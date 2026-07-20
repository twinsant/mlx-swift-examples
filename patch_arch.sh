#!/bin/bash
FILE="/Users/twinsant/Library/Developer/Xcode/DerivedData/mlx-swift-examples-doihvqomapcsjtgajqwervkdfnea/SourcePackages/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/device.cpp"
sed -i '' 's/arch_ = std::string(device_->architecture()->name()->utf8String());/auto arch = device_ ? device_->architecture() : nullptr;\n    auto name = arch ? arch->name() : nullptr;\n    auto c_str = name ? name->utf8String() : nullptr;\n    arch_ = c_str ? std::string(c_str) : "unknown";/g' "$FILE"
