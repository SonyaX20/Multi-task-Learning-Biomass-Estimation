# Multitask Learning Architecture Diagram Guide

This document provides detailed specifications for creating architecture diagrams for the TreeSatAI-CHM and iMAESTRO multitask learning models.

---

## 1. TreeSatAI-CHM MultiTaskUNet Architecture

### Overview
- **Input**: 4-channel SAR image (VV, VH, VV/VH ratio, CHM) at 6×6 pixels
- **Tasks**: 
  - Classification: Patch-level genus classification (15 classes)
  - Regression: Pixel-level height prediction (6×6 output)
- **Architecture**: Hard parameter sharing with shared encoder, dual decoder heads

---

### Layer-by-Layer Specification (Left to Right)

#### **INPUT LAYER**
```
Box: "Input"
Size: (4, 6, 6)
Channels: 4 (VV, VH, Ratio, CHM)
Spatial: 6×6 pixels
Color: Light blue
```

**Arrow → Encoder Stage 1**
- Operation: DoubleConv (Conv3×3 + BN + ReLU) × 2
- Label: "DoubleConv\n3→B channels"

---

#### **ENCODER STAGE 1**
```
Box: "Encoder 1"
Size: (B, 6, 6)
Channels: B (base channels, typically 32 or 64)
Spatial: 6×6 pixels
Color: Green
Operations within:
  - Conv 3×3, padding=1
  - BatchNorm
  - ReLU
  - Conv 3×3, padding=1
  - BatchNorm
  - ReLU
```

**Arrow → Encoder Stage 2 (Downsampling)**
- Operation: Strided Conv3×3, stride=2
- Label: "Down\nStride Conv\n6×6→3×3\nB→2B channels"
- Size change: (B, 6, 6) → (2B, 3, 3)

**Skip Connection (dashed line) → Decoder Stage 1**
- Label: "Skip\nConcat"
- Carries: (B, 6, 6) features

---

#### **ENCODER STAGE 2**
```
Box: "Encoder 2"
Size: (2B, 3, 3)
Channels: 2B
Spatial: 3×3 pixels
Color: Green (darker)
Operations within:
  - Conv 3×3, padding=1
  - BatchNorm
  - ReLU
  - Conv 3×3, padding=1
  - BatchNorm
  - ReLU
```

**Arrow → Bottleneck (Downsampling)**
- Operation: Strided Conv3×3, stride=3
- Label: "Down\nStride Conv\n3×3→1×1\n2B→4B channels"
- Size change: (2B, 3, 3) → (4B, 1, 1)

**Skip Connection (dashed line) → Decoder Stage 2**
- Label: "Skip\nConcat"
- Carries: (2B, 3, 3) features

---

#### **BOTTLENECK (Shared Feature Space)**
```
Box: "Bottleneck"
Size: (4B, 1, 1)
Channels: 4B
Spatial: 1×1 pixel
Color: Orange (highlight as shared)
Operations within:
  - Conv 3×3, padding=1
  - BatchNorm
  - ReLU
  - Conv 3×3, padding=1
  - BatchNorm
  - ReLU
```

**BRANCHING POINT** - Two paths diverge here:

---

### **PATH 1: CLASSIFICATION HEAD**

**Arrow → Classification Branch**
- Operation: Global Average Pooling
- Label: "GAP\n(4B, 1, 1)→(4B,)"
- Size change: (4B, 1, 1) → (4B,)

#### **Classification MLP**
```
Box: "Classification Head"
Size: (4B,) → (15,)
Color: Purple
Operations within:
  - Flatten: (4B, 1, 1) → (4B,)
  - Linear: 4B → 128
  - ReLU
  - Dropout(0.3)
  - Linear: 128 → 15
```

**Arrow → Classification Output**
- Operation: None (logits)
- Label: "Logits"

#### **CLASSIFICATION OUTPUT**
```
Box: "Class Logits"
Size: (15,)
Shape: 15 genus classes
Color: Purple (light)
```

**Arrow → Classification Loss**
- Operation: Binary Cross-Entropy with Logits
- Label: "BCE Loss"

#### **CLASSIFICATION LOSS**
```
Box: "L_cls"
Formula: BCE(pred, target)
Color: Red
Details:
  - Multi-label BCE
  - Class weights: w_c = 1/freq_c
  - Normalized by running mean
```

---

### **PATH 2: REGRESSION HEAD**

**Arrow → Decoder Stage 2 (Upsampling)**
- Operation: Transposed Conv, stride=3
- Label: "Up\nTransConv\n1×1→3×3\n4B→2B channels"
- Size change: (4B, 1, 1) → (2B, 3, 3)

#### **DECODER STAGE 2**
```
Box: "Decoder 2"
Size: (2B, 3, 3) + Skip(2B, 3, 3) → (4B, 3, 3)
Channels: 4B (after concat)
Spatial: 3×3 pixels
Color: Blue
Operations within:
  - Concatenate with skip: (2B, 3, 3) + (2B, 3, 3) = (4B, 3, 3)
  - DoubleConv: Conv3×3 + BN + ReLU (×2)
  - Output: (2B, 3, 3)
```

**Arrow → Decoder Stage 1 (Upsampling)**
- Operation: Transposed Conv, stride=2
- Label: "Up\nTransConv\n3×3→6×6\n2B→B channels"
- Size change: (2B, 3, 3) → (B, 6, 6)

#### **DECODER STAGE 1**
```
Box: "Decoder 1"
Size: (B, 6, 6) + Skip(B, 6, 6) → (2B, 6, 6)
Channels: 2B (after concat)
Spatial: 6×6 pixels
Color: Blue (lighter)
Operations within:
  - Concatenate with skip: (B, 6, 6) + (B, 6, 6) = (2B, 6, 6)
  - DoubleConv: Conv3×3 + BN + ReLU (×2)
  - Dropout(0.3)
  - Output: (B, 6, 6)
```

**Arrow → Regression Output**
- Operation: Conv1×1 + Softplus
- Label: "Conv1×1\nSoftplus"
- Size change: (B, 6, 6) → (1, 6, 6)

#### **REGRESSION OUTPUT**
```
Box: "Height Map"
Size: (1, 6, 6)
Shape: 36 height values
Color: Blue (light)
Activation: Softplus (ensures non-negative)
```

**Arrow → Regression Loss**
- Operation: Root Mean Squared Error
- Label: "RMSE Loss"

#### **REGRESSION LOSS**
```
Box: "L_reg"
Formula: sqrt(mean((pred - target)²))
Color: Red
Details:
  - Computed per pixel
  - Normalized by running mean
```

---

### **LOSS COMBINATION**

Both losses feed into:

#### **LOSS NORMALIZATION**
```
Box: "Loss Normalization"
Color: Orange
Operations:
  L̃_cls = L_cls / L̄_cls
  L̃_reg = L_reg / L̄_reg
  
Where L̄ is running mean (momentum=0.9)
```

**Arrow → Total Loss**

#### **UNCERTAINTY WEIGHTING (Optional)**
```
Box: "Uncertainty Weighting"
Color: Orange
Formula:
  L_total = (1/(2σ²_cls)) × L̃_cls + (1/2)log(σ²_cls)
          + (1/(2σ²_reg)) × L̃_reg + (1/2)log(σ²_reg)
  
Learnable parameters: σ_cls, σ_reg
```

**OR**

#### **FIXED WEIGHTING**
```
Box: "Fixed Weighting"
Color: Orange
Formula:
  L_total = w_cls × L̃_cls + w_reg × L̃_reg
  
Hyperparameters: w_cls, w_reg
```

**Arrow → Optimizer**

#### **FINAL LOSS**
```
Box: "L_total"
Color: Red (bold)
Feeds to: Optimizer (Adam)
Backpropagates through entire network
```

---

### Gradient Flow Annotations

Add these as annotations on the diagram:

1. **From L_cls**: Gradients flow through Classification Head → Bottleneck → Encoder
2. **From L_reg**: Gradients flow through Decoder → Bottleneck → Encoder
3. **Gradient Conflict**: At Bottleneck, gradients from both tasks meet
   - Measure: Cosine similarity between ∇_cls and ∇_reg
   - Negative cosine = conflicting gradients

---

### Visual Design Guidelines

**Colors:**
- Input: Light blue (#E3F2FD)
- Encoder: Green gradient (#C8E6C9 → #66BB6A)
- Bottleneck: Orange (#FFB74D) - highlight as shared
- Classification path: Purple (#CE93D8 → #AB47BC)
- Regression path: Blue (#90CAF9 → #42A5F5)
- Loss boxes: Red (#EF5350)
- Loss processing: Orange (#FF9800)

**Arrows:**
- Solid arrows: Data flow
- Dashed arrows: Skip connections
- Bold arrows: Loss backpropagation
- Color-coded by path (purple for cls, blue for reg)

**Box Styling:**
- Rounded corners
- Drop shadow for depth
- Bold border for bottleneck
- Size proportional to feature dimensions

**Text Annotations:**
- Layer name (bold)
- Tensor shape: (C, H, W)
- Operations inside box
- Arrow labels for transformations

---

## 2. iMAESTRO Multi-Task U-Net Architecture

### Overview
- **Input**: 2-channel SAR image (VV, VH) at 64×64 pixels
- **Tasks**: 
  - Segmentation: Pixel-level genus segmentation (7 classes)
  - Height Regression: Pixel-level height prediction
  - Biomass Regression: Pixel-level biomass prediction
- **Architecture**: Hard parameter sharing with shared encoder, three independent decoder branches

---

### Layer-by-Layer Specification (Left to Right)

#### **INPUT LAYER**
```
Box: "Input"
Size: (2, 64, 64)
Channels: 2 (VV, VH)
Spatial: 64×64 pixels
Color: Light blue
```

**Arrow → Encoder Stage 1**
- Operation: DoubleConv (Conv3×3 + BN + ReLU) × 2
- Label: "DoubleConv\n2→B channels"

---

#### **ENCODER STAGE 1**
```
Box: "Encoder 1"
Size: (B, 64, 64)
Channels: B (base channels, typically 32)
Spatial: 64×64 pixels
Color: Green
Operations within:
  - Conv 3×3, padding=1
  - BatchNorm
  - ReLU
  - Conv 3×3, padding=1
  - BatchNorm
  - ReLU
```

**Arrow → Encoder Stage 2 (Downsampling)**
- Operation: Strided Conv3×3, stride=2
- Label: "Down\nStride Conv\n64×64→32×32\nB→2B channels"
- Size change: (B, 64, 64) → (2B, 32, 32)

**Skip Connection (dashed line) → All 3 Decoder Stage 1**
- Label: "Skip\nConcat"
- Carries: (B, 64, 64) features
- Branches to 3 decoders

---

#### **ENCODER STAGE 2**
```
Box: "Encoder 2"
Size: (2B, 32, 32)
Channels: 2B
Spatial: 32×32 pixels
Color: Green (darker)
Operations within:
  - Conv 3×3, padding=1
  - BatchNorm
  - ReLU
  - Conv 3×3, padding=1
  - BatchNorm
  - ReLU
```

**Arrow → Encoder Stage 3 (Downsampling)**
- Operation: Strided Conv3×3, stride=2
- Label: "Down\nStride Conv\n32×32→16×16\n2B→4B channels"
- Size change: (2B, 32, 32) → (4B, 16, 16)

**Skip Connection (dashed line) → All 3 Decoder Stage 2**
- Label: "Skip\nConcat"
- Carries: (2B, 32, 32) features
- Branches to 3 decoders

---

#### **ENCODER STAGE 3**
```
Box: "Encoder 3"
Size: (4B, 16, 16)
Channels: 4B
Spatial: 16×16 pixels
Color: Green (even darker)
Operations within:
  - Conv 3×3, padding=1
  - BatchNorm
  - ReLU
  - Conv 3×3, padding=1
  - BatchNorm
  - ReLU
```

**Arrow → Encoder Stage 4 (Downsampling)**
- Operation: Strided Conv3×3, stride=2
- Label: "Down\nStride Conv\n16×16→8×8\n4B→8B channels"
- Size change: (4B, 16, 16) → (8B, 8, 8)

**Skip Connection (dashed line) → All 3 Decoder Stage 3**
- Label: "Skip\nConcat"
- Carries: (4B, 16, 16) features
- Branches to 3 decoders

---

#### **ENCODER STAGE 4**
```
Box: "Encoder 4"
Size: (8B, 8, 8)
Channels: 8B
Spatial: 8×8 pixels
Color: Green (darkest)
Operations within:
  - Conv 3×3, padding=1
  - BatchNorm
  - ReLU
  - Conv 3×3, padding=1
  - BatchNorm
  - ReLU
```

**Arrow → Bottleneck (Downsampling)**
- Operation: Strided Conv3×3, stride=2
- Label: "Down\nStride Conv\n8×8→4×4\n8B→16B channels"
- Size change: (8B, 8, 8) → (16B, 4, 4)

**Skip Connection (dashed line) → All 3 Decoder Stage 4**
- Label: "Skip\nConcat"
- Carries: (8B, 8, 8) features
- Branches to 3 decoders

---

#### **BOTTLENECK (Shared Feature Space)**
```
Box: "Bottleneck"
Size: (16B, 4, 4)
Channels: 16B
Spatial: 4×4 pixels
Color: Orange (highlight as shared)
Operations within:
  - Conv 3×3, padding=1
  - BatchNorm
  - ReLU
  - Conv 3×3, padding=1
  - BatchNorm
  - ReLU
```

**BRANCHING POINT** - Three paths diverge here:

---

### **PATH 1: SEGMENTATION DECODER**

**Arrow → Seg Decoder Stage 4 (Upsampling)**
- Operation: Transposed Conv, stride=2
- Label: "Up\nTransConv\n4×4→8×8\n16B→8B channels"
- Size change: (16B, 4, 4) → (8B, 8, 8)

#### **SEG DECODER STAGE 4**
```
Box: "Seg Dec 4"
Size: (8B, 8, 8) + Skip(8B, 8, 8) → (16B, 8, 8)
Channels: 16B (after concat)
Spatial: 8×8 pixels
Color: Purple
Operations within:
  - Concatenate with skip: (8B, 8, 8) + (8B, 8, 8) = (16B, 8, 8)
  - DoubleConv: Conv3×3 + BN + ReLU (×2)
  - Output: (8B, 8, 8)
```

**Arrow → Seg Decoder Stage 3 (Upsampling)**
- Operation: Transposed Conv, stride=2
- Label: "Up\nTransConv\n8×8→16×16\n8B→4B channels"

#### **SEG DECODER STAGE 3**
```
Box: "Seg Dec 3"
Size: (4B, 16, 16) + Skip(4B, 16, 16) → (8B, 16, 16)
Channels: 8B (after concat)
Spatial: 16×16 pixels
Color: Purple
Operations within:
  - Concatenate with skip
  - DoubleConv
  - Output: (4B, 16, 16)
```

**Arrow → Seg Decoder Stage 2 (Upsampling)**
- Operation: Transposed Conv, stride=2
- Label: "Up\nTransConv\n16×16→32×32\n4B→2B channels"

#### **SEG DECODER STAGE 2**
```
Box: "Seg Dec 2"
Size: (2B, 32, 32) + Skip(2B, 32, 32) → (4B, 32, 32)
Channels: 4B (after concat)
Spatial: 32×32 pixels
Color: Purple
Operations within:
  - Concatenate with skip
  - DoubleConv
  - Output: (2B, 32, 32)
```

**Arrow → Seg Decoder Stage 1 (Upsampling)**
- Operation: Transposed Conv, stride=2
- Label: "Up\nTransConv\n32×32→64×64\n2B→B channels"

#### **SEG DECODER STAGE 1**
```
Box: "Seg Dec 1"
Size: (B, 64, 64) + Skip(B, 64, 64) → (2B, 64, 64)
Channels: 2B (after concat)
Spatial: 64×64 pixels
Color: Purple
Operations within:
  - Concatenate with skip
  - DoubleConv
  - Dropout(0.4)
  - Output: (128, 64, 64)
```

**Arrow → Segmentation Output**
- Operation: Conv1×1
- Label: "Conv1×1\n128→7"
- Size change: (128, 64, 64) → (7, 64, 64)

#### **SEGMENTATION OUTPUT**
```
Box: "Genus Logits"
Size: (7, 64, 64)
Shape: 7 classes × 4096 pixels
Color: Purple (light)
```

**Arrow → Segmentation Loss**
- Operation: Cross-Entropy (masked)
- Label: "CE Loss\n(masked)"

#### **SEGMENTATION LOSS**
```
Box: "L_seg"
Formula: CE(pred, target, mask)
Color: Red
Details:
  - Masked cross-entropy
  - Ignores 16 rare classes
  - Only 6 dominant genera
  - Normalized by running mean
```

---

### **PATH 2: HEIGHT REGRESSION DECODER**

**Arrow → Height Decoder Stage 4 (Upsampling)**
- Operation: Transposed Conv, stride=2
- Label: "Up\nTransConv\n4×4→8×8\n16B→8B channels"

#### **HEIGHT DECODER STAGE 4**
```
Box: "Height Dec 4"
Size: (8B, 8, 8) + Skip(8B, 8, 8) → (16B, 8, 8)
Channels: 16B (after concat)
Spatial: 8×8 pixels
Color: Blue
Operations within:
  - Concatenate with skip
  - DoubleConv
  - Output: (8B, 8, 8)
```

**Arrow → Height Decoder Stage 3 (Upsampling)**
- Operation: Transposed Conv, stride=2
- Label: "Up\nTransConv\n8×8→16×16\n8B→4B channels"

#### **HEIGHT DECODER STAGE 3**
```
Box: "Height Dec 3"
Size: (4B, 16, 16) + Skip(4B, 16, 16) → (8B, 16, 16)
Channels: 8B (after concat)
Spatial: 16×16 pixels
Color: Blue
Operations within:
  - Concatenate with skip
  - DoubleConv
  - Output: (4B, 16, 16)
```

**Arrow → Height Decoder Stage 2 (Upsampling)**
- Operation: Transposed Conv, stride=2
- Label: "Up\nTransConv\n16×16→32×32\n4B→2B channels"

#### **HEIGHT DECODER STAGE 2**
```
Box: "Height Dec 2"
Size: (2B, 32, 32) + Skip(2B, 32, 32) → (4B, 32, 32)
Channels: 4B (after concat)
Spatial: 32×32 pixels
Color: Blue
Operations within:
  - Concatenate with skip
  - DoubleConv
  - Output: (2B, 32, 32)
```

**Arrow → Height Decoder Stage 1 (Upsampling)**
- Operation: Transposed Conv, stride=2
- Label: "Up\nTransConv\n32×32→64×64\n2B→B channels"

#### **HEIGHT DECODER STAGE 1**
```
Box: "Height Dec 1"
Size: (B, 64, 64) + Skip(B, 64, 64) → (2B, 64, 64)
Channels: 2B (after concat)
Spatial: 64×64 pixels
Color: Blue
Operations within:
  - Concatenate with skip
  - DoubleConv
  - Dropout(0.2)
  - Output: (128, 64, 64)
```

**Arrow → Height Output**
- Operation: Conv1×1
- Label: "Conv1×1\n128→1"
- Size change: (128, 64, 64) → (1, 64, 64)

#### **HEIGHT OUTPUT**
```
Box: "Height Map"
Size: (1, 64, 64)
Shape: 4096 height values
Color: Blue (light)
Activation: Linear (masked to valid pixels)
```

**Arrow → Height Loss**
- Operation: RMSE (masked)
- Label: "RMSE Loss\n(masked)"

#### **HEIGHT LOSS**
```
Box: "L_height"
Formula: sqrt(mean((pred - target)² × mask))
Color: Red
Details:
  - Masked RMSE
  - Only valid forest pixels
  - Normalized by running mean
```

---

### **PATH 3: BIOMASS REGRESSION DECODER**

**Arrow → Biomass Decoder Stage 4 (Upsampling)**
- Operation: Transposed Conv, stride=2
- Label: "Up\nTransConv\n4×4→8×8\n16B→8B channels"

#### **BIOMASS DECODER STAGE 4**
```
Box: "Biomass Dec 4"
Size: (8B, 8, 8) + Skip(8B, 8, 8) → (16B, 8, 8)
Channels: 16B (after concat)
Spatial: 8×8 pixels
Color: Cyan
Operations within:
  - Concatenate with skip
  - DoubleConv
  - Output: (8B, 8, 8)
```

**Arrow → Biomass Decoder Stage 3 (Upsampling)**
- Operation: Transposed Conv, stride=2
- Label: "Up\nTransConv\n8×8→16×16\n8B→4B channels"

#### **BIOMASS DECODER STAGE 3**
```
Box: "Biomass Dec 3"
Size: (4B, 16, 16) + Skip(4B, 16, 16) → (8B, 16, 16)
Channels: 8B (after concat)
Spatial: 16×16 pixels
Color: Cyan
Operations within:
  - Concatenate with skip
  - DoubleConv
  - Output: (4B, 16, 16)
```

**Arrow → Biomass Decoder Stage 2 (Upsampling)**
- Operation: Transposed Conv, stride=2
- Label: "Up\nTransConv\n16×16→32×32\n4B→2B channels"

#### **BIOMASS DECODER STAGE 2**
```
Box: "Biomass Dec 2"
Size: (2B, 32, 32) + Skip(2B, 32, 32) → (4B, 32, 32)
Channels: 4B (after concat)
Spatial: 32×32 pixels
Color: Cyan
Operations within:
  - Concatenate with skip
  - DoubleConv
  - Output: (2B, 32, 32)
```

**Arrow → Biomass Decoder Stage 1 (Upsampling)**
- Operation: Transposed Conv, stride=2
- Label: "Up\nTransConv\n32×32→64×64\n2B→B channels"

#### **BIOMASS DECODER STAGE 1**
```
Box: "Biomass Dec 1"
Size: (B, 64, 64) + Skip(B, 64, 64) → (2B, 64, 64)
Channels: 2B (after concat)
Spatial: 64×64 pixels
Color: Cyan
Operations within:
  - Concatenate with skip
  - DoubleConv
  - Dropout(0.2)
  - Output: (128, 64, 64)
```

**Arrow → Biomass Output**
- Operation: Conv1×1
- Label: "Conv1×1\n128→1"
- Size change: (128, 64, 64) → (1, 64, 64)

#### **BIOMASS OUTPUT**
```
Box: "Biomass Map"
Size: (1, 64, 64)
Shape: 4096 biomass values
Color: Cyan (light)
Activation: Linear (masked to valid pixels)
```

**Arrow → Biomass Loss**
- Operation: RMSE (masked)
- Label: "RMSE Loss\n(masked)"

#### **BIOMASS LOSS**
```
Box: "L_biomass"
Formula: sqrt(mean((pred - target)² × mask))
Color: Red
Details:
  - Masked RMSE
  - Only valid forest pixels
  - Normalized by running mean
```

---

### **ALLOMETRIC CONSTRAINT (Optional)**

Add a connection between Height Output and Biomass Output:

#### **ALLOMETRIC LOSS**
```
Box: "L_allom"
Formula: mean((log(B) - α - β×log(H))²)
Color: Orange
Details:
  - Couples height and biomass
  - α = 0.0673, β = 2.5
  - Physics-based constraint
  - Weight: λ_allom
```

**Arrows:**
- From Height Output → L_allom
- From Biomass Output → L_allom
- From L_allom → Total Loss

---

### **LOSS COMBINATION**

All three losses feed into:

#### **LOSS NORMALIZATION**
```
Box: "Loss Normalization"
Color: Orange
Operations:
  L̃_seg = L_seg / L̄_seg
  L̃_height = L_height / L̄_height
  L̃_biomass = L_biomass / L̄_biomass
  
Where L̄ is running mean (momentum=0.9)
```

**Arrow → Total Loss**

#### **UNCERTAINTY WEIGHTING**
```
Box: "Uncertainty Weighting"
Color: Orange
Formula:
  L_total = (1/(2σ²_seg)) × L̃_seg + (1/2)log(σ²_seg)
          + (1/(2σ²_height)) × L̃_height + (1/2)log(σ²_height)
          + (1/(2σ²_biomass)) × L̃_biomass + (1/2)log(σ²_biomass)
          + λ_allom × L_allom
  
Learnable parameters: σ_seg, σ_height, σ_biomass
Hyperparameter: λ_allom
```

**Arrow → Optimizer**

#### **FINAL LOSS**
```
Box: "L_total"
Color: Red (bold)
Feeds to: Optimizer (Adam)
Backpropagates through entire network
```

---

### Gradient Flow Annotations

Add these as annotations on the diagram:

1. **From L_seg**: Gradients flow through Seg Decoder → Bottleneck → Encoder
2. **From L_height**: Gradients flow through Height Decoder → Bottleneck → Encoder
3. **From L_biomass**: Gradients flow through Biomass Decoder → Bottleneck → Encoder
4. **From L_allom**: Gradients flow to both Height and Biomass outputs
5. **Gradient Conflict**: At Bottleneck, gradients from all three tasks meet
   - Measure: Pairwise cosine similarity between task gradients
   - Track: seg-height, seg-biomass, height-biomass

---

### Visual Design Guidelines

**Colors:**
- Input: Light blue (#E3F2FD)
- Encoder: Green gradient (#C8E6C9 → #1B5E20)
- Bottleneck: Orange (#FFB74D) - highlight as shared
- Segmentation path: Purple (#CE93D8 → #AB47BC)
- Height path: Blue (#90CAF9 → #42A5F5)
- Biomass path: Cyan (#80DEEA → #00ACC1)
- Loss boxes: Red (#EF5350)
- Loss processing: Orange (#FF9800)
- Allometric constraint: Orange (#FFA726)

**Arrows:**
- Solid arrows: Data flow
- Dashed arrows: Skip connections (branch to all 3 decoders)
- Bold arrows: Loss backpropagation
- Color-coded by path (purple/blue/cyan)
- Orange dashed: Allometric constraint

**Box Styling:**
- Rounded corners
- Drop shadow for depth
- Bold border for bottleneck
- Size proportional to feature dimensions
- Three parallel decoder columns

**Text Annotations:**
- Layer name (bold)
- Tensor shape: (C, H, W)
- Operations inside box
- Arrow labels for transformations

**Layout:**
- Encoder: Vertical stack on left
- Bottleneck: Center, highlighted
- Three decoders: Parallel columns on right
- Losses: Bottom row, aligned with outputs
- Loss combination: Far right

---

## 3. Implementation Tips for Drawing Tools

### For PowerPoint/Keynote:
1. Use SmartArt for basic flow
2. Custom shapes for layers
3. Connectors for arrows
4. Text boxes for annotations
5. Group related elements
6. Use transparency for skip connections

### For draw.io/Lucidchart:
1. Use rectangle shapes with rounded corners
2. Use arrow connectors with labels
3. Use colors from palette above
4. Use layers for organization
5. Use groups for complex blocks
6. Export as SVG for LaTeX

### For TikZ (LaTeX):
1. Use `\node` for boxes
2. Use `\draw` for arrows
3. Use `positioning` library for layout
4. Use `fit` for grouping
5. Use `decorations` for skip connections
6. See example code below

### For Python (matplotlib/graphviz):
1. Use `matplotlib.patches.FancyBboxPatch`
2. Use `matplotlib.patches.FancyArrowPatch`
3. Use `networkx` for layout
4. Use `graphviz` for automatic layout
5. Export as PDF for LaTeX

---

## 4. Key Differences Between Models

| Aspect | TreeSatAI-CHM | iMAESTRO |
|--------|---------------|----------|
| Input size | 6×6 pixels | 64×64 pixels |
| Input channels | 4 (VV, VH, Ratio, CHM) | 2 (VV, VH) |
| Encoder depth | 2 stages | 4 stages |
| Bottleneck size | 1×1 | 4×4 |
| Number of tasks | 2 (classification, regression) | 3 (segmentation, 2 regressions) |
| Classification type | Patch-level (15 classes) | Pixel-level (7 classes) |
| Regression output | Height only | Height + Biomass |
| Skip connections | 2 levels | 4 levels |
| Decoder branches | 2 (cls head + reg decoder) | 3 (full decoders) |
| Special constraints | None | Allometric loss (optional) |
| Loss masking | No | Yes (forest mask) |

---

## 5. Critical Details to Include

### Both Models:
1. **Tensor shapes at every layer** - (C, H, W) format
2. **Operations on arrows** - Conv, TransConv, GAP, etc.
3. **Skip connection paths** - Dashed lines with "Concat" labels
4. **Activation functions** - ReLU, Softplus, etc.
5. **Normalization** - BatchNorm locations
6. **Dropout** - Locations and rates
7. **Loss formulas** - Mathematical notation
8. **Gradient flow** - Backpropagation paths
9. **Shared vs. task-specific** - Color coding
10. **Parameter counts** - Optional annotations

### TreeSatAI-CHM Specific:
- Global Average Pooling for classification
- Asymmetric architecture (head vs. decoder)
- Softplus activation for height
- Class weighting for imbalanced data

### iMAESTRO Specific:
- Three parallel decoders
- Masked loss functions
- Allometric constraint between tasks
- Higher dropout for segmentation (0.4 vs. 0.2)
- Larger spatial dimensions throughout

---

## 6. Suggested Layout Dimensions

### TreeSatAI-CHM:
- Width: 1200-1500 pixels
- Height: 800-1000 pixels
- Encoder column: 20% width
- Bottleneck: 10% width
- Classification branch: 30% width (top)
- Regression branch: 30% width (bottom)
- Loss section: 10% width (right)

### iMAESTRO:
- Width: 1800-2000 pixels
- Height: 1000-1200 pixels
- Encoder column: 15% width
- Bottleneck: 10% width
- Three decoder columns: 15% each (45% total)
- Loss section: 20% width (right)

---

## 7. Example Annotations

Add these text boxes near relevant sections:

**Near Bottleneck:**
```
"Shared Feature Space
All tasks learn from
common representations"
```

**Near Skip Connections:**
```
"Skip connections preserve
spatial details lost during
downsampling"
```

**Near Loss Combination:**
```
"Uncertainty weighting
automatically balances
task contributions"
```

**Near Gradient Flow:**
```
"Gradients from all tasks
meet at bottleneck
Potential for conflict"
```

---

This guide provides complete specifications for creating detailed MTL architecture diagrams. Use it as a reference when drawing the diagrams in your preferred tool.
