We initially implemented the ViT base model on the CIFAR-10 dataset, following the framework established by.
The transformer architecture comprised 12 layers, each with 10 heads, and an embedding dimension of 768. 
Training this model on a Google Colab Notebook required approximately 45 minutes per epoch. To optimize our approach, we implemented a low-rank version of ViT without decompositions, 
reducing the training time to an average of 30 minutes per epoch on the CIFAR-10 dataset. We created a low-rank approximation with the help of three classes. 
a low-rank-linear class, followed by a low-rank-self-attention class, and finally, a low-rank-transformer-block class. The low-rank linear class is created by adding two linear layers.
A linear layer computes Y = XW + b, where X is the input, W is the weight matrix, and $b$ is the bias. The low-rank approximation comes from decomposing the weight matrix W into smaller matrices, 
which will reduce the number of parameters. This approach is done using PyTorch's \texttt{nn.Linear} that maps the input from its original dimension to a lower dimension. The second layer then maps the reduced 
dimension back up to the desired output dimension. To reduce the parameter count to improve the computational efficiency, we add a self-attention class that uses query, key, and value projection coming from the 
two low-rank layers. After computing QKV matrices, we reshape them for multi-head attention, and then change dimensions to align for multiplication. For attention computation, we take the dot product of queries
and keys, scale it, and then apply softmax to it to get the attention weights. The final attended output is computed by multiplying the weights with values. Finally, we will have a transformer block that has 
two feed-forward mlp layers and normalization and attention mechanism.   After 10 epochs, this model achieved an accuracy of 59.75\%. Subsequently, we applied Singular Value Decomposition (SVD) to the low-rank 
ViT model to further decrease the training time. However, this adjustment resulted in a decrease in accuracy to 45.77\% after 10 epochs on the same dataset.
