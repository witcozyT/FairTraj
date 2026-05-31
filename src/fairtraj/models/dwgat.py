import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class DensityAwareGATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=2, dropout=0.1):
        super().__init__()
        self.gat = GATConv(in_channels, out_channels, heads=heads, dropout=dropout, concat=True)
        self.yita = 0.5

    def forward(self, x, edge_index, density):
        """
        x: 节点特征 [node_num, feature_dim]
        edge_index: 边索引 [2, edge_num]
        density: 节点密度 [node_num, 1]
        """
        # 第一层不进行密度聚合
        if x.shape[-1] != 2:
            # 根据密度聚合
            node_from, node_to = edge_index
            density_from = torch.exp(density[node_from])
            denom = torch.zeros(x.shape[0], 1, device=x.device).scatter_add_(0, node_to.unsqueeze(-1), density_from)
            beta = density_from / (denom[node_to] + 1e-16)
            x_dw = torch.zeros_like(x).scatter_add_(0, node_to.unsqueeze(-1).expand(-1, x.size(1)), beta * x[node_from])
        else:
            x_dw = None

        # 根据特征聚合
        x_gat = self.gat(x, edge_index)  # [node_num, out_channels * heads]

        if x_dw != None:
            # 加性特征融合
            x = self.yita * x_gat + (1 - self.yita) * x_dw
        else:
            x = x_gat

        return x


class DensityAwareGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()

        # 构建多层GAT
        self.layers.append(DensityAwareGATLayer(in_channels, hidden_channels, heads))
        for _ in range(num_layers - 1):
            self.layers.append(DensityAwareGATLayer(hidden_channels * heads, hidden_channels, heads))

        self.final_dim = hidden_channels * heads * num_layers
        self.encoder = nn.Linear(self.final_dim, hidden_channels)
        self.out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, density):
        layer_outputs = []
        for layer in self.layers:
            x = layer(x, edge_index, density)
            layer_outputs.append(x)

        h_final = torch.cat(layer_outputs, dim=1) # 多尺度聚合
        hidden = self.encoder(h_final)
        pred = self.out(hidden)

        return hidden, pred

def train(data, save_path=None, epochs=3000, lr=0.002, log_every=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = data.to(device)
    density = data.x[:, 0:1]

    model = DensityAwareGAT(in_channels=2, hidden_channels=128, out_channels=1, heads=3)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        _, density_recon = model(data.x[:, 1:], data.edge_index, density)
        loss = F.mse_loss(density_recon, density)
        loss.backward()
        optimizer.step()
        if log_every and (epoch + 1) % log_every == 0:
            print(f"Epoch {epoch}, Loss {loss.item():.4f}")

    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    return model

def eval(model, data, save_path=None, save_fig=True):
    model.eval()
    with torch.no_grad():
        x = data.x[:, 1:]
        edge_index = data.edge_index
        density = data.x[:, 0]
        embeddings, _ = model(x, edge_index, density.unsqueeze(1))

    embeddings = embeddings.cpu().numpy()

    if save_fig:
        density = density.cpu().numpy()

        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        sc = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=density,
            cmap='viridis',
            alpha=0.6,
            s=10
        )
        plt.colorbar(sc, label='Node Density')
        plt.title('Node Embeddings Colored by Density')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    return embeddings
