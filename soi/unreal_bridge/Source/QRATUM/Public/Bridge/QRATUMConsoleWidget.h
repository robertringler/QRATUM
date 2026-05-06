// QRATUMConsoleWidget.h — v1.2 real-time console UMG widget.
// Owner: Agents 1+5+7. Bridges typed text -> `console_send` over the existing
// launcher socket; renders inbound `console_intent|ack|result|error|log`
// frames in a multiline log.
//
// CRITICAL: this widget never executes intents directly; every command
// flows launcher -> BROKER -> IntentClient -> UQRATUMCommandRouter.

#pragma once

#include "CoreMinimal.h"
#include "Blueprint/UserWidget.h"
#include "QRATUMConsoleWidget.generated.h"

class UMultiLineEditableTextBox;
class UEditableTextBox;

UCLASS(Blueprintable, BlueprintType)
class QRATUM_API UQRATUMConsoleWidget : public UUserWidget
{
    GENERATED_BODY()

public:
    virtual void NativeConstruct() override;
    virtual void NativeDestruct() override;

    /** Append a line to the output log (game-thread safe). */
    UFUNCTION(BlueprintCallable, Category = "QRATUM|Console")
    void AppendOutput(const FString& Line);

    /** Submit raw text as a console command. Called by Enter handler. */
    UFUNCTION(BlueprintCallable, Category = "QRATUM|Console")
    void SubmitRaw(const FString& Raw);

protected:
    UFUNCTION()
    void HandleInputCommitted(const FText& Text, ETextCommit::Type CommitMethod);

    UPROPERTY(meta = (BindWidget))
    UMultiLineEditableTextBox* OutputLog = nullptr;

    UPROPERTY(meta = (BindWidget))
    UEditableTextBox* InputLine = nullptr;

    /** Maximum scrollback lines kept in OutputLog. */
    UPROPERTY(EditAnywhere, Category = "QRATUM|Console")
    int32 MaxScrollback = 1024;

private:
    int32 LineCount = 0;
    FString Buffer;
    FDelegateHandle FrameSubscription;
};
