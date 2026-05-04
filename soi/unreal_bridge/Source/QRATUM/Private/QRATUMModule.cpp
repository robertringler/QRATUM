// QRATUMModule.cpp
// Module entry point for the QRATUM swarm runtime.

#include "Modules/ModuleManager.h"

class FQRATUMModule : public IModuleInterface
{
public:
    virtual void StartupModule() override {}
    virtual void ShutdownModule() override {}
};

IMPLEMENT_MODULE(FQRATUMModule, QRATUM);
