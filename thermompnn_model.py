import torch
import torch.nn as nn
from protein_mpnn_utils import ProteinMPNN, featurize


def load_protein_mpnn(cfg):
    """Loading Pre-trained ProteinMPNN model for structure embeddings"""

    checkpoint = torch.load(cfg.platform.proteinmpnn_weights, map_location='cuda')
    
    if cfg.platform.model_type == "protein_mpnn":
        model = ProteinMPNN(num_letters=cfg.proteinmpnn_model.num_letters, 
                        node_features=cfg.proteinmpnn_model.hidden_dim, 
                        edge_features=cfg.proteinmpnn_model.hidden_dim, 
                        hidden_dim=cfg.proteinmpnn_model.hidden_dim, 
                        num_encoder_layers=cfg.proteinmpnn_model.num_layers, 
                        num_decoder_layers=cfg.proteinmpnn_model.num_layers, 
                        k_neighbors=checkpoint['num_edges'], 
                        augment_eps=checkpoint['noise_level'], model_type="protein_mpnn",
                        device='cuda')

    elif cfg.platform.model_type.startswith("ligand_mpnn"):
        model = ProteinMPNN(num_letters=cfg.proteinmpnn_model.num_letters, 
                        node_features=cfg.proteinmpnn_model.hidden_dim, 
                        edge_features=cfg.proteinmpnn_model.hidden_dim, 
                        hidden_dim=cfg.proteinmpnn_model.hidden_dim, 
                        num_encoder_layers=cfg.proteinmpnn_model.num_layers, 
                        num_decoder_layers=cfg.proteinmpnn_model.num_layers, 
                        k_neighbors=checkpoint['num_edges'], atom_context_num=checkpoint['atom_context_num'], 
                        augment_eps=checkpoint['noise_level'], model_type=cfg.platform.model_type,
                        device='cuda')

    if cfg.model.load_pretrained:
        model.load_state_dict(checkpoint['model_state_dict'])

    model.to('cuda')

    if cfg.model.freeze_weights: # freeze these weights for transfer learning
        model.eval()
        if cfg.platform.model_type.startswith("ligand_mpnn"):
            protein_ligand_graph_weights = ['W_e.weight', 'W_e.bias', 'encoder_layers.0.norm1.weight', 'encoder_layers.0.norm1.bias', 'encoder_layers.0.norm2.weight', 'encoder_layers.0.norm2.bias', 'encoder_layers.0.norm3.weight', 'encoder_layers.0.norm3.bias', 'encoder_layers.0.W1.weight', 'encoder_layers.0.W1.bias', 'encoder_layers.0.W2.weight', 'encoder_layers.0.W2.bias', 'encoder_layers.0.W3.weight', 'encoder_layers.0.W3.bias', 'encoder_layers.0.W11.weight', 'encoder_layers.0.W11.bias', 'encoder_layers.0.W12.weight', 'encoder_layers.0.W12.bias', 'encoder_layers.0.W13.weight', 'encoder_layers.0.W13.bias', 'encoder_layers.0.dense.W_in.weight', 'encoder_layers.0.dense.W_in.bias', 'encoder_layers.0.dense.W_out.weight', 'encoder_layers.0.dense.W_out.bias', 'encoder_layers.1.norm1.weight', 'encoder_layers.1.norm1.bias', 'encoder_layers.1.norm2.weight', 'encoder_layers.1.norm2.bias', 'encoder_layers.1.norm3.weight', 'encoder_layers.1.norm3.bias', 'encoder_layers.1.W1.weight', 'encoder_layers.1.W1.bias', 'encoder_layers.1.W2.weight', 'encoder_layers.1.W2.bias', 'encoder_layers.1.W3.weight', 'encoder_layers.1.W3.bias', 'encoder_layers.1.W11.weight', 'encoder_layers.1.W11.bias', 'encoder_layers.1.W12.weight', 'encoder_layers.1.W12.bias', 'encoder_layers.1.W13.weight', 'encoder_layers.1.W13.bias', 'encoder_layers.1.dense.W_in.weight', 'encoder_layers.1.dense.W_in.bias', 'encoder_layers.1.dense.W_out.weight', 'encoder_layers.1.dense.W_out.bias', 'encoder_layers.2.norm1.weight', 'encoder_layers.2.norm1.bias', 'encoder_layers.2.norm2.weight', 'encoder_layers.2.norm2.bias', 'encoder_layers.2.norm3.weight', 'encoder_layers.2.norm3.bias', 'encoder_layers.2.W1.weight', 'encoder_layers.2.W1.bias', 'encoder_layers.2.W2.weight', 'encoder_layers.2.W2.bias', 'encoder_layers.2.W3.weight', 'encoder_layers.2.W3.bias', 'encoder_layers.2.W11.weight', 'encoder_layers.2.W11.bias', 'encoder_layers.2.W12.weight', 'encoder_layers.2.W12.bias', 'encoder_layers.2.W13.weight', 'encoder_layers.2.W13.bias', 'encoder_layers.2.dense.W_in.weight', 'encoder_layers.2.dense.W_in.bias', 'encoder_layers.2.dense.W_out.weight', 'encoder_layers.2.dense.W_out.bias', 'features.embeddings.linear.weight', 'features.embeddings.linear.bias', 'features.edge_embedding.weight', 'features.norm_edges.weight', 'features.norm_edges.bias', 'features.node_project_down.weight', 'features.node_project_down.bias', 'features.norm_nodes.weight', 'features.norm_nodes.bias', 'features.type_linear.weight', 'features.type_linear.bias', 'features.y_nodes.weight', 'features.y_edges.weight', 'features.norm_y_edges.weight', 'features.norm_y_edges.bias', 'features.norm_y_nodes.weight', 'features.norm_y_nodes.bias', 'W_s.weight', 'decoder_layers.0.norm1.weight', 'decoder_layers.0.norm1.bias', 'decoder_layers.0.norm2.weight', 'decoder_layers.0.norm2.bias', 'decoder_layers.0.W1.weight', 'decoder_layers.0.W1.bias', 'decoder_layers.0.W2.weight', 'decoder_layers.0.W2.bias', 'decoder_layers.0.W3.weight', 'decoder_layers.0.W3.bias', 'decoder_layers.0.dense.W_in.weight', 'decoder_layers.0.dense.W_in.bias', 'decoder_layers.0.dense.W_out.weight', 'decoder_layers.0.dense.W_out.bias', 'decoder_layers.1.norm1.weight', 'decoder_layers.1.norm1.bias', 'decoder_layers.1.norm2.weight', 'decoder_layers.1.norm2.bias', 'decoder_layers.1.W1.weight', 'decoder_layers.1.W1.bias', 'decoder_layers.1.W2.weight', 'decoder_layers.1.W2.bias', 'decoder_layers.1.W3.weight', 'decoder_layers.1.W3.bias', 'decoder_layers.1.dense.W_in.weight', 'decoder_layers.1.dense.W_in.bias', 'decoder_layers.1.dense.W_out.weight', 'decoder_layers.1.dense.W_out.bias', 'decoder_layers.2.norm1.weight', 'decoder_layers.2.norm1.bias', 'decoder_layers.2.norm2.weight', 'decoder_layers.2.norm2.bias', 'decoder_layers.2.W1.weight', 'decoder_layers.2.W1.bias', 'decoder_layers.2.W2.weight', 'decoder_layers.2.W2.bias', 'decoder_layers.2.W3.weight', 'decoder_layers.2.W3.bias', 'decoder_layers.2.dense.W_in.weight', 'decoder_layers.2.dense.W_in.bias', 'decoder_layers.2.dense.W_out.weight', 'decoder_layers.2.dense.W_out.bias', 'W_out.weight', 'W_out.bias']
            for key, param in model.state_dict().items():
                if key in protein_ligand_graph_weights:
                    param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = False

    return model


class TransferModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dims = list(cfg.model.hidden_dims)
        self.subtract_mut = cfg.model.subtract_mut
        self.num_final_layers = cfg.model.num_final_layers
        self.lightattn = cfg.model.lightattn if 'lightattn' in cfg.model else False

        if 'decoding_order' not in self.cfg:
            self.cfg.decoding_order = 'left-to-right'
        
        self.prot_mpnn = load_protein_mpnn(cfg)

        hid_sizes = [ cfg.proteinmpnn_model.hidden_dim*(self.num_final_layers+1) ]
        hid_sizes += self.hidden_dims
        hid_sizes += [ cfg.proteinmpnn_model.num_letters ]

        print('MLP HIDDEN SIZES:', hid_sizes)

        if self.lightattn:
            print('Enabled LightAttention')
            self.light_attention = LightAttention(embeddings_dim=cfg.proteinmpnn_model.hidden_dim*(self.num_final_layers+1))

        self.both_out = nn.Sequential()

        for sz1, sz2 in zip(hid_sizes, hid_sizes[1:]):
            self.both_out.append(nn.ReLU())
            self.both_out.append(nn.Linear(sz1, sz2))

        self.ddg_out = nn.Linear(1, 1)

    def fwd(self, mpnn_embed, mpnn_hid, mut, mask):
        inputs = []
        if self.num_final_layers > 0:
            hid = mpnn_hid[0][mut.position]  # MPNN hidden embeddings at mut position
            inputs.append(hid)

        embed = mpnn_embed[0][mut.position]  # MPNN seq embeddings at mut position
        inputs.append(embed)

        # concatenating hidden layers and embeddings
        lin_input = torch.cat(inputs, -1)

        # passing vector through lightattn
        if self.lightattn:
            lin_input = torch.unsqueeze(torch.unsqueeze(lin_input, -1), 0)
            lin_input = self.light_attention(lin_input, mask)

        both_input = torch.unsqueeze(self.both_out(lin_input), -1)
        dG_out = self.ddg_out(both_input)

        return dG_out

    def forward(self, pdb, mutations):        
        protein_dict = featurize(pdb[0], number_of_ligand_atoms=16, model_type=self.cfg.platform.model_type)
        complex = mutations[0].complex
        mpnn_embed, mpnn_hid, _, mpnn_hid_ctxt, _ = self.prot_mpnn(protein_dict, complex=complex)
        if self.num_final_layers > 0:
            if mpnn_hid is not None:
                mpnn_hid = torch.cat(mpnn_hid[:self.num_final_layers], -1)
            if mpnn_hid_ctxt is not None:
                mpnn_hid_ctxt = torch.cat(mpnn_hid_ctxt[:self.num_final_layers], -1)

        out = []
        for mut in mutations:

            if mut is None:
                out.append(None)
                continue

            assert mut.complex == complex

            aa_index = 'ACDEFGHIKLMNPQRSTVWYX'.index(mut.mutation)
            wt_aa_index = 'ACDEFGHIKLMNPQRSTVWYX'.index(mut.wildtype)

            ddG = 0
            if mpnn_hid is not None:
                dG_out = self.fwd(mpnn_embed, mpnn_hid, mut, protein_dict["mask"])
                if self.subtract_mut:
                    ddG = dG_out[aa_index][0] - dG_out[wt_aa_index][0]
                else:
                    ddG = dG_out[aa_index][0]
            if mpnn_hid_ctxt is not None:
                dG_out_ctxt = self.fwd(mpnn_embed, mpnn_hid_ctxt, mut, protein_dict["mask"])
                if self.subtract_mut:
                    ddG = dG_out_ctxt[aa_index][0] - dG_out_ctxt[wt_aa_index][0] - ddG
                else:
                    ddG = dG_out_ctxt[aa_index][0] - ddG

            out.append({
                "ddG": torch.unsqueeze(ddG, 0),
            })
        return out, None


class LightAttention(nn.Module):
    """Source:
    Hannes Stark et al. 2022
    https://github.com/HannesStark/protein-localization/blob/master/models/light_attention.py
    """
    def __init__(self, embeddings_dim=1024, output_dim=11, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25):
        super(LightAttention, self).__init__()

        self.feature_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                             padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                               padding=kernel_size // 2)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(conv_dropout)

    def forward(self, x: torch.Tensor, mask, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.
        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        
        o = self.dropout(o)  # [batch_gsize, embeddings_dim, sequence_length]

        attention = self.attention_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        
        o1 = o * self.softmax(attention)
        return torch.squeeze(o1)
