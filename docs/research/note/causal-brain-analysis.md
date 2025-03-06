
```mermaid
flowchart TD
    A[Input MRI Image] --> B[Preprocessing<br/>(Crop Brain, Resize, Normalize)]
    B --> C[CVAE UNet Encoder]
    C --> D[Feature Maps<br/>(x1, x2, x3, x4, x5)]
    D --> E[Global Avg Pooling<br/>(on x5)]
    E --> F[Concatenate One-hot Label]
    F --> G[FC Layers<br/>(Produce $\mu$, $\log\sigma^2$)]
    G --> H[Reparameterization<br/>(Sample $z$)]
    H --> I[CVAE UNet Decoder<br/>(Uses Skip Connections from x4, x3, x2, x1)]
    I --> J[Reconstructed Image]

    H --- K[Latent Space $z$]
    K --- L[Deep SVDD Loss<br/>(Minimize $\|z - c\|^2$)]
    K --- M[Local Smoothness Loss<br/>(Minimize distance among same-class $z$)]
    
    J --> N[Output: Reconstructed Image]
    
    subgraph Losses
      O[Reconstruction Loss (MSE)]
      P[KL Divergence Loss]
    end

    J --- O
    G --- P
```