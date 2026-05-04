#pragma once

#include "CoreMinimal.h"
#include "UObject/Interface.h"
#include "QRATUMHealthProvider.generated.h"

UINTERFACE(BlueprintType)
class UQRATUMHealthProvider : public UInterface
{
	GENERATED_BODY()
};

// Pure C++ interface contract.
// Implement on any Pawn to supply normalized health (0.0-1.0) to the Brain.
// If not implemented, Brain defaults to 1.0f.
class IQRATUMHealthProvider
{
	GENERATED_BODY()

public:

	virtual float GetNormalizedHealth() const = 0;
};
