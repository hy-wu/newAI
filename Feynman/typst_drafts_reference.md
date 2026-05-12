# Typst Drafts Reference (Preserved before LaTeX migration)

These are the section skeleton drafts originally written in Typst, saved for reference.

## 1.1 The Gap Between Math and Code

Current deep learning practice is driven by empirical experimentation, while theoretical understanding lags behind. This gap between the mathematical structures underlying neural networks and the imperative code that implements them creates friction in design, optimization, and reasoning about model behavior. We argue that diagrammatic formalisms — inspired by physics — can bridge this gap by providing a unified visual language that is simultaneously mathematically rigorous and computationally actionable.

## 1.2 The Diagrammatic Vision / Feynman Diagram Vision for AI

We introduce the central thesis: Feynman diagrams and related diagrammatic formalisms offer a revolutionary way to represent, analyze, compile, and optimize neural network architectures. By mapping external lines to tokens, vertices to attention operations, and loops to recurrence, we obtain a unified representation language that spans from high-level architecture design down to GPU kernel compilation. This vision draws together ideas from quantum field theory, category theory, statistical physics, and compiler design into a coherent framework for next-generation AI infrastructure.

## 1.3 Scope and Outline

This survey covers the interdisciplinary foundations needed to realize the diagrammatic vision: physical mathematics (QFT, path integrals, geometric algebra, p-adic numbers), neural network theory (field theories, NTK, scaling laws), attention mechanisms (path integral formulation, statistical physics), and compilation infrastructure (ML IRs, tensor compilation, equality saturation, architecture search). We then synthesize these threads into the vision of a unified Feynman Diagram Compiler for AI, identifying open problems and future directions.

## 2.1 Feynman Diagrams and Quantum Field Theory

We review the fundamentals of Feynman diagrams as a perturbative expansion tool in quantum field theory, establishing the physical intuition that will be mapped to neural network computations. Key concepts include: external lines as particles (inputs/outputs), vertices as interactions (nonlinear operations), loops as virtual corrections (recurrence), and the diagrammatic expansion as a systematic way to compute observables. The connection between Feynman diagrams and computational graphs is drawn through the work of Peskin and Schroeder (QFT) and modern reinterpretations of diagrams as a form of compositional computation.

## 2.2 Tensor Networks and Diagrammatic Algebra

Tensor networks provide a diagrammatic language for factorizing high-dimensional tensors into networks of contracted lower-order tensors, with deep connections to quantum many-body physics and machine learning. We survey the key tensor network architectures — matrix product states (MPS), projected entangled pair states (PEPS), and tree tensor networks — and their applications in ML, including tensorized embeddings and attention approximations. The diagrammatic calculus of tensor networks serves as a bridge between the physics notion of entanglement and the ML notion of representational capacity.

## 2.3 Diagrammatic Languages for Model Architecture

Beyond tensor networks, a broader family of diagrammatic formalisms has emerged for specifying and reasoning about model architectures. We survey string diagrams from category theory (used in DisCoPy for NLP), neural circuit diagrams, and block-diagram DSLs found in ML frameworks. These languages differ in expressive power and compositionality, ranging from informal visual notations to fully formal categorical calculi with equational reasoning capabilities.

## 2.4 The Physical Mathematician's Toolbox

We introduce three additional mathematical structures that enrich the diagrammatic framework. (1) Geometric (Clifford) algebra provides a unified language for representing geometric transformations, rotations, and equivariances in neural networks, with applications in 3D data and Lorentz-equivariant transformers. (2) p-adic numbers and ultrametric geometry naturally capture hierarchical linguistic structure via ultrametric attention with O(N log N) complexity. (3) Density matrices from quantum mechanics offer a formalism for representing uncertainty and ambiguity in language models, where ambiguous tokens are treated as mixed states that collapse to definite meanings given context.

## 3.1 Neural Networks as Field Theories

We review the mapping between neural networks and quantum/statistical field theories. In the infinite-width limit, neural networks with i.i.d. parameters become Gaussian processes (NNGP), while gradient descent dynamics map to the neural tangent kernel (NTK). Finite-width corrections introduce interactions that are captured by higher-order Feynman diagrams. We cover the large-N expansion as a systematic perturbation theory for neural networks, the effective field theory (EFT) approach to scaling laws, and the renormalization group perspective on depth and representation learning.

## 3.2 Diagrammatic Theory of Deep Neural Networks

This section develops the diagrammatic representation of neural network computations as Feynman-like diagrams. Each layer corresponds to a vertex connecting input and output lines, with nonlinearities acting as interaction vertices. Backpropagation maps to the Hermitian conjugate diagram, creating a direct analogy between gradient computation and cutting/gluing operations in diagrammatic perturbation theory. We show how this perspective unifies forward and backward passes into a single diagrammatic language.

## 3.3 Model Architectures as Feynman Diagrams

We examine specific neural network architectures through the lens of Feynman diagrams. Convolutional networks correspond to locally-connected vertices with weight sharing; transformers combine pairwise attention vertices with skip connections; mixture-of-experts introduces conditional routing vertices. Each architectural motif maps to a distinct diagrammatic pattern, and architectural innovation becomes a process of composing and mutating diagram fragments rather than engineering imperative code.

## 3.4 Field-Theoretic Perspectives: From Feynman Diagrams to Computational Graphs

Building on the foundations, we examine the deeper theoretical implications of treating computational graphs as Feynman diagrams. This perspective suggests new approaches to architecture design through diagrammatic operations and invariants — analogous to how Feynman diagrams reveal symmetries and conservation laws in physics. We discuss the potential for discovering new architectures through diagrammatic reasoning, automatic perturbation theory for neural networks, and the relationship between diagram topology and computational complexity.

## 4.1 Path Integral Formulation of Attention and Transformers

The path integral formulation of quantum mechanics provides a natural language for attention mechanisms, where the attention score emerges from summing over all possible pairwise token interaction paths. We review recent work on integral transformers and folded context condensation, which cast transformer forward passes as path integrals over token trajectories. This perspective reveals attention as a mechanism for computing transition amplitudes in a learned metric space, with the softmax operation playing the role of the Boltzmann factor.

## 4.2 Attention as Statistical Physics

We develop the statistical physics interpretation of attention: the softmax function is the maximum-entropy (exponential family) distribution over token interactions, with temperature controlling the sharpness of attention. The attention matrix becomes a partition function, and the self-attention layer computes a form of free energy minimization. This section connects to energy-based models, the principle of maximum entropy, and the role of temperature scaling in modern LLMs, providing a unified thermodynamic picture.

## 5.1 Intermediate Representations for Deep Learning

We survey the landscape of intermediate representations (IRs) for deep learning compilers: computational graphs (graph IRs), MLIR with its dialect system, Relay (TVM's high-level IR), Tiramisu's polyhedral representation, and Relax's dynamic shape abstractions. Each IR occupies a different point in the expressiveness-compilability tradeoff space. We analyze how an ideal diagrammatic IR would combine the compositionality of categorical string diagrams with the optimization power of MLIR's multi-level lowering infrastructure.

## 5.2 Compilation and Auto-Optimization of Tensor Programs

Modern DL compilers (TVM, Halide, XLA, Glow, Hidet) decompose the compilation process into operator scheduling, memory planning, and code generation, with auto-tuning (Ansor, OpenTuner, AutoTVM) searching the optimization space. We review the core techniques: tensor expression lowering, kernel fusion, memory hierarchy optimization, and auto-scheduling. We assess how these techniques map onto the diagrammatic compilation vision, where diagram fragments correspond to kernel launch configurations and schedule decisions.

## 5.3 Graph Rewriting, Superoptimization, and Equality Saturation

Graph rewriting and equality saturation (egg, Metatheory.jl) offer powerful tools for optimizing computational graphs through exhaustive equivalence search. We review the e-graph data structure, equality saturation algorithm, and its application to tensor program optimization (DialEgg, eqsat, Mirage). We argue that diagrammatic representations are naturally suited for equality saturation because Feynman diagram identities (Wick contraction, crossing symmetry, optical theorem) correspond directly to tensor graph rewrite rules.

## 5.4 Neural Architecture Search and Differentiable Programming

Neural architecture search (NAS) automates model design through search over architectural motifs. We review the evolution from RL-based NAS (Zoph & Le) through differentiable approaches (DARTS, ENAS, ProxylessNAS) to hardware-aware search (MnasNet, FBNet). We argue that a diagrammatic grammar — where architectures are generated by composing diagram fragments — provides a more structured and interpretable search space than current cell-based or graph-based approaches, enabling principled architecture generation guided by diagrammatic invariants.

## 6.1 Compiling Diagrammatic Representations to GPU

We explore the compilation pipeline from high-level diagrammatic representations to efficient GPU code. Key challenges include mapping diagram fragments to tensor operations, handling dynamic graph structures (attention masks, conditional computation), and exploiting hardware-specific optimizations (tensor cores, memory hierarchy). We review work on GPU compilation for sparse patterns (EC-SpMM), concurrent tensor programs (ConCo), and Triton-style tile-level programming as promising building blocks.

## 6.2 A Unified Diagrammatic Compiler for AI

We synthesize the survey into the vision of a Feynman Diagram Compiler: a system that takes architecture-as-diagram as input and produces optimized GPU code through a pipeline of diagrammatic analysis, rewriting, and lowering. Key components include: (1) a diagrammatic IR with formal equational theory, (2) a library of diagram rewrite rules derived from physical identities, (3) categorical semantics connecting diagrams to tensor operations, and (4) compilation strategies informed by diagram topology. Existing categorical frameworks (DisCoPy, neural circuit diagrams) and multi-level IRs (MLIR, Relax) serve as starting points. This section concludes with a concrete proposal and research roadmap.

## 6.3 Cross-Domain Analogies as Inductive Biases

The diagrammatic perspective naturally suggests cross-domain analogies: Monte Carlo methods inform stochastic computation graphs, computational fluid dynamics maps to information flow in VLMs, and large-scale structure formation in cosmology parallels latent variable inference in generative models. We survey these analogies as potential sources of inductive biases for architecture design, arguing that the diagrammatic language makes such transfers systematic by providing a shared formal grammar across domains.

## 7.1 Open Problems, Benchmarks, and Outlook

We identify open problems: (1) developing a complete categorical semantics for transformer architectures, (2) proving correctness of diagram rewrite rules for tensor programs, (3) scaling equality saturation to production-scale models, (4) designing benchmark suites for diagrammatic compilation, and (5) the theoretical limitations of diagrammatic representations. We discuss how progress in any of these areas would advance the unified compiler vision.

## 7.2 Conclusion: The Diagram is the Code

We conclude by restating the central vision: the Feynman diagram is not merely a theoretical tool for understanding neural networks, but a practical foundation for the next generation of AI infrastructure. Just as Feynman diagrams revolutionized the practice of quantum field theory by making complex calculations manageable and intuitive, a diagrammatic foundation for deep learning can transform how we design, analyze, compile, and optimize neural networks — making the diagram itself the code.
