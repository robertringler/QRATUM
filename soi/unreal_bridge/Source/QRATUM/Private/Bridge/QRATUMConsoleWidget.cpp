// QRATUMConsoleWidget.cpp — v1.2 console widget (UI only).
// Owner: Agents 1, 7. The widget renders the local scrollback and forwards
// committed text to UQRATUMBridgeSubsystem::SendConsoleLine() which owns
// the actual console socket lifecycle (v2.1 wiring).

#include "Bridge/QRATUMConsoleWidget.h"
#include "Components/MultiLineEditableTextBox.h"
#include "Components/EditableTextBox.h"
#include "Bridge/QRATUMBridgeSubsystem.h"
#include "Engine/World.h"

void UQRATUMConsoleWidget::NativeConstruct()
{
    Super::NativeConstruct();

    if (InputLine)
    {
        InputLine->OnTextCommitted.AddDynamic(this,
            &UQRATUMConsoleWidget::HandleInputCommitted);
        InputLine->SetKeyboardFocus();
    }
    AppendOutput(TEXT("[qratum] console attached. type :help for examples."));
}

void UQRATUMConsoleWidget::NativeDestruct()
{
    if (InputLine)
    {
        InputLine->OnTextCommitted.RemoveDynamic(this,
            &UQRATUMConsoleWidget::HandleInputCommitted);
    }
    Super::NativeDestruct();
}

void UQRATUMConsoleWidget::HandleInputCommitted(const FText& Text,
                                                ETextCommit::Type CommitMethod)
{
    if (CommitMethod != ETextCommit::OnEnter)
    {
        return;
    }
    const FString Raw = Text.ToString();
    if (Raw.IsEmpty())
    {
        return;
    }
    AppendOutput(FString::Printf(TEXT("> %s"), *Raw));
    if (InputLine)
    {
        InputLine->SetText(FText::GetEmpty());
        InputLine->SetKeyboardFocus();
    }
    SubmitRaw(Raw);
}

void UQRATUMConsoleWidget::SubmitRaw(const FString& Raw)
{
    if (UWorld* World = GetWorld())
    {
        if (UQRATUMBridgeSubsystem* Bridge =
                World->GetSubsystem<UQRATUMBridgeSubsystem>())
        {
            // v2.1: Bridge owns a dedicated console-socket connection
            // and forwards "console_send" frames. Until that ships, the
            // widget logs the attempt locally so the UI loop can be
            // exercised in standalone mode.
            const FString Frame = FString::Printf(
                TEXT("{\"cmd\":\"console_send\",\"raw\":\"%s\"}"),
                *Raw.ReplaceCharWithEscapedChar());
            // TODO(v2.1): Bridge->SendConsoleFrame(Frame);
            (void)Frame;
        }
    }
}

void UQRATUMConsoleWidget::AppendOutput(const FString& Line)
{
    if (!OutputLog)
    {
        return;
    }
    Buffer += Line;
    Buffer += TEXT("\n");
    LineCount++;
    if (LineCount > MaxScrollback)
    {
        // trim oldest line
        int32 NL = INDEX_NONE;
        if (Buffer.FindChar(TEXT('\n'), NL) && NL >= 0 && NL + 1 < Buffer.Len())
        {
            Buffer = Buffer.Mid(NL + 1);
            LineCount--;
        }
    }
    OutputLog->SetText(FText::FromString(Buffer));
}
