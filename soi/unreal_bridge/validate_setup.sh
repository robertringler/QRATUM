# validate_setup.sh (updated for IntentOS)

# ... (previous top-of-file checks remain unchanged)

# Check Unreal files
if [ -f "$SCRIPT_DIR/IntentOS.uproject" ]; then
    success "UE5 project file found"
else
    error "UE5 project file missing"
fi

if [ -f "$SCRIPT_DIR/Source/IntentOS/Public/SoiTelemetrySubsystem.h" ]; then
    success "C++ header found"
else
    error "C++ header missing"
fi

if [ -f "$SCRIPT_DIR/Source/IntentOS/Private/SoiTelemetrySubsystem.cpp" ]; then
    success "C++ implementation found"
else
    error "C++ implementation missing"
fi

if [ -f "$SCRIPT_DIR/Source/IntentOS/IntentOS.Build.cs" ]; then
    success "UE5 Build.cs found"
else
    error "UE5 Build.cs missing"
fi

# ... (rest of file left unchanged)
