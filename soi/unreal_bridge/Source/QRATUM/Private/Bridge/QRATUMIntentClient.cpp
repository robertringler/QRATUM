// QRATUMIntentClient.cpp — socket transport thread.
//
// Resilience: connection drops are non-fatal — the thread sleeps and
// retries with exponential backoff up to 5 s. Stop() cleanly tears down.

#include "Bridge/QRATUMIntentClient.h"
#include "Sockets.h"
#include "SocketSubsystem.h"
#include "Common/TcpSocketBuilder.h"
#include "IPAddress.h"
#include "HAL/RunnableThread.h"
#include "HAL/PlatformProcess.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Dom/JsonObject.h"

namespace
{
    constexpr int32 kRecvBufferBytes = 64 * 1024;
}

FQRATUMIntentClient::FQRATUMIntentClient(const FString& InHost, int32 InPort)
    : Host(InHost)
    , Port(InPort)
{
    Thread = FRunnableThread::Create(this, TEXT("QRATUM.IntentClient"), 0, TPri_BelowNormal);
}

FQRATUMIntentClient::~FQRATUMIntentClient()
{
    Stop();
    if (Thread)
    {
        Thread->WaitForCompletion();
        delete Thread;
        Thread = nullptr;
    }
    DisconnectSocket();
}

bool FQRATUMIntentClient::Init()
{
    return true;
}

void FQRATUMIntentClient::Stop()
{
    bStop = true;
}

void FQRATUMIntentClient::Exit() { }

bool FQRATUMIntentClient::ConnectOnce()
{
    ISocketSubsystem* SocketSubsystem = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
    if (!SocketSubsystem) return false;

    DisconnectSocket();

    Socket = FTcpSocketBuilder(TEXT("QRATUM.Intent"))
        .AsBlocking()
        .AsReusable()
        .Build();
    if (!Socket) return false;

    TSharedRef<FInternetAddr> Addr = SocketSubsystem->CreateInternetAddr();
    bool bIsValid = false;
    Addr->SetIp(*Host, bIsValid);
    Addr->SetPort(Port);
    if (!bIsValid)
    {
        UE_LOG(LogTemp, Warning, TEXT("[QRATUM/Intent] invalid host %s"), *Host);
        DisconnectSocket();
        return false;
    }

    if (!Socket->Connect(*Addr))
    {
        DisconnectSocket();
        return false;
    }

    // Send v1.0 subscribe handshake.
    const FString Handshake = TEXT("{\"cmd\":\"subscribe\",\"v\":\"1.0\"}\n");
    const FTCHARToUTF8 Conv(*Handshake);
    int32 Sent = 0;
    {
        FScopeLock Lk(&SocketWriteCS);
        Socket->Send(reinterpret_cast<const uint8*>(Conv.Get()), Conv.Length(), Sent);
    }

    bConnected = true;
    UE_LOG(LogTemp, Log, TEXT("[QRATUM/Intent] connected %s:%d"), *Host, Port);
    return true;
}

void FQRATUMIntentClient::DisconnectSocket()
{
    bConnected = false;
    if (Socket)
    {
        Socket->Close();
        ISocketSubsystem* SocketSubsystem = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM);
        if (SocketSubsystem)
        {
            SocketSubsystem->DestroySocket(Socket);
        }
        Socket = nullptr;
    }
}

bool FQRATUMIntentClient::ParseIntentLine(const FString& Line, FQRATUMIntent& Out) const
{
    if (Line.IsEmpty()) return false;

    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Line);
    TSharedPtr<FJsonObject> Obj;
    if (!FJsonSerializer::Deserialize(Reader, Obj) || !Obj.IsValid())
    {
        UE_LOG(LogTemp, Warning, TEXT("[QRATUM/Intent] non-JSON line dropped: %s"), *Line);
        return false;
    }

    // Tolerate either a bare intent or {"type":"intent","name":..,"params":..}.
    FString TypeStr;
    Obj->TryGetStringField(TEXT("type"), TypeStr);
    // Drop heartbeats silently — they only refresh staleness on the launcher.
    if (TypeStr.ToLower() == TEXT("heartbeat"))
    {
        return false;
    }
    if (!TypeStr.IsEmpty() && TypeStr.ToLower() != TEXT("intent"))
    {
        return false;
    }

    if (!Obj->TryGetStringField(TEXT("name"), Out.Name) || Out.Name.IsEmpty())
    {
        return false;
    }
    Obj->TryGetStringField(TEXT("id"), Out.Id);
    int64 Epoch = 0;
    if (Obj->TryGetNumberField(TEXT("epoch"), Epoch))
    {
        Out.Epoch = Epoch;
    }
    const TSharedPtr<FJsonObject>* ParamsObj = nullptr;
    if (Obj->TryGetObjectField(TEXT("params"), ParamsObj) && ParamsObj && ParamsObj->IsValid())
    {
        Out.Params = *ParamsObj;
    }
    Out.RawJson = Line;
    return true;
}

uint32 FQRATUMIntentClient::Run()
{
    int32 BackoffMs = 250;
    TArray<uint8> Recv;
    Recv.SetNumUninitialized(kRecvBufferBytes);

    while (!bStop)
    {
        if (!ConnectOnce())
        {
            FPlatformProcess::Sleep(BackoffMs / 1000.0f);
            BackoffMs = FMath::Min(BackoffMs * 2, 5000);
            continue;
        }
        BackoffMs = 250;

        while (!bStop && Socket)
        {
            // 1. Drain pending outbound frames (ack/result/etc.)
            FString OutLine;
            while (OutQueue.Dequeue(OutLine))
            {
                if (!OutLine.EndsWith(TEXT("\n"))) OutLine.AppendChar(TEXT('\n'));
                const FTCHARToUTF8 Bytes(*OutLine);
                int32 Wrote = 0;
                FScopeLock Lk(&SocketWriteCS);
                if (!Socket || !Socket->Send(reinterpret_cast<const uint8*>(Bytes.Get()),
                                             Bytes.Length(), Wrote))
                {
                    // re-enqueue then break to reconnect
                    OutQueue.Enqueue(OutLine);
                    break;
                }
            }

            uint32 Pending = 0;
            if (!Socket || !Socket->HasPendingData(Pending))
            {
                FPlatformProcess::Sleep(0.01f);
                // probe for half-open connections.
                int32 Probe = 0;
                uint8 Tiny[1];
                if (Socket->Recv(Tiny, 0, Probe, ESocketReceiveFlags::Peek))
                {
                    // ok
                }
                continue;
            }

            int32 Read = 0;
            if (!Socket->Recv(Recv.GetData(), Recv.Num(), Read) || Read <= 0)
            {
                break; // disconnected
            }

            // append to read buffer as UTF-8; split on newline.
            TArray<ANSICHAR> NullTerm;
            NullTerm.Append(reinterpret_cast<const ANSICHAR*>(Recv.GetData()), Read);
            NullTerm.Add('\0');
            ReadBuffer += UTF8_TO_TCHAR(NullTerm.GetData());
            int32 NewlineIdx = INDEX_NONE;
            while (ReadBuffer.FindChar(TEXT('\n'), NewlineIdx))
            {
                FString Line = ReadBuffer.Left(NewlineIdx).TrimStartAndEnd();
                ReadBuffer.RemoveAt(0, NewlineIdx + 1);

                FQRATUMIntent Intent;
                if (ParseIntentLine(Line, Intent))
                {
                    Queue.Enqueue(Intent);
                    ReceivedCount.Increment();
                }
            }
        }

        DisconnectSocket();
        if (!bStop)
        {
            FPlatformProcess::Sleep(0.5f);
        }
    }

    DisconnectSocket();
    return 0;
}

bool FQRATUMIntentClient::PopIntent(FQRATUMIntent& Out)
{
    return Queue.Dequeue(Out);
}

bool FQRATUMIntentClient::SendLine(const FString& Line)
{
    if (!bConnected || Line.IsEmpty()) return false;
    OutQueue.Enqueue(Line);
    return true;
}
