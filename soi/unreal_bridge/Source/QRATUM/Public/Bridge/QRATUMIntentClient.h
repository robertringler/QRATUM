// QRATUMIntentClient.h — Socket client transport.
// Runs on its OWN thread. Connects to launcher (default 127.0.0.1:5555),
// sends a {"cmd":"subscribe"} handshake, then reads newline-delimited JSON
// lines. Every parsed intent is enqueued into a thread-safe MPSC queue
// drained on the game thread by UQRATUMBridgeSubsystem::Tick.
//
// CRITICAL: this class never touches gameplay objects. It is a pure
// transport layer.

#pragma once

#include "CoreMinimal.h"
#include "HAL/Runnable.h"
#include "HAL/ThreadSafeBool.h"
#include "Containers/Queue.h"
#include "Bridge/QRATUMIntent.h"

class FSocket;
class FRunnableThread;

/**
 * Background socket-reader thread. Owns the FSocket lifecycle.
 * Push raw JSON lines via Pop(); writer is the worker thread, reader is the
 * game thread (single-consumer / single-producer).
 */
class QRATUM_API FQRATUMIntentClient : public FRunnable
{
public:
    FQRATUMIntentClient(const FString& InHost, int32 InPort);
    virtual ~FQRATUMIntentClient() override;

    // FRunnable
    virtual bool   Init() override;
    virtual uint32 Run() override;
    virtual void   Stop() override;
    virtual void   Exit() override;

    /** Game-thread: dequeue one intent. Returns false if empty. */
    bool PopIntent(FQRATUMIntent& Out);

    /**
     * Thread-safe outbound send. Appends '\n' if missing. Used by the
     * router to ship ack/result frames back to the launcher.
     * Returns false if not currently connected (caller may drop or retry).
     */
    bool SendLine(const FString& Line);

    /** Connection state (atomic). */
    bool IsConnected() const { return bConnected; }

    /** Total intents successfully parsed since start. */
    int64 GetReceivedCount() const { return ReceivedCount.GetValue(); }

private:
    bool ConnectOnce();
    void DisconnectSocket();
    bool ParseIntentLine(const FString& Line, FQRATUMIntent& Out) const;

    FString             Host;
    int32               Port;
    FSocket*            Socket = nullptr;
    FRunnableThread*    Thread = nullptr;
    FThreadSafeBool     bStop;
    FThreadSafeBool     bConnected;
    FThreadSafeCounter64 ReceivedCount;

    TQueue<FQRATUMIntent, EQueueMode::Spsc> Queue;
    FString             ReadBuffer;

    /** Outbound queue (Mpsc: many game-thread writers, single socket reader). */
    TQueue<FString, EQueueMode::Mpsc> OutQueue;
    FCriticalSection    SocketWriteCS;
};
