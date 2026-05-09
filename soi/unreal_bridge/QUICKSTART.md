# Open: soi/unreal_bridge/IntentOS.uproject

# 4. Follow the Blueprint guide
# Read: soi/unreal_bridge/BLUEPRINT_IMPLEMENTATION_GUIDE.md

## Prerequisites (5 minutes)

### 1. Install Rust
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### 2. Install Unreal Engine 5.7.4+
- Download Epic Games Launcher: https://www.unrealengine.com/
- Install Unreal Engine 5.7.4 or later

### 3. Install C++ Build Tools

**Windows:**
- Visual Studio 2022 with "Desktop development with C++"

**macOS:**
```bash
xcode-select --install
```

**Linux:**
```bash
sudo apt-get install build-essential clang lld
```

## Step-by-Step Setup (5 minutes)

### Step 1: Validate Setup

```bash
cd soi/unreal_bridge
./validate_setup.sh
```

Expected output:
```
✓ Rust installed
✓ Cargo installed
✓ C++ compiler found
✓ All files present
✓ Rust code compiles
✓ Tests pass
```
