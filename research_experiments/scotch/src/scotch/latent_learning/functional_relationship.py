"""This file implements the functional relationship required for SCOTCH"""

import torch
from causica.functional_relationships.icgnn import FGNNI, generate_fully_connected
from causica.functional_relationships.icgnn import (
    ICGNN as DECIEmbedFunctionalRelationships,
)
from torch import nn


class SCOTCHFunctionalRelationships(DECIEmbedFunctionalRelationships):
    """This adapts from DECIEmbedFunctionalRelationships for SCOTCH by incorporting sigmoid output activation, res_connection and norm_layer"""

    def __init__(
        self,
        shapes: dict[str, torch.Size],
        embedding_size: int | None = None,
        out_dim_g: int | None = None,
        norm_layer: type[nn.LayerNorm] | None = None,
        res_connection: bool = False,
        sigmoid_output: bool = False,
    ):
        super().__init__(
            shapes=shapes,
            embedding_size=embedding_size,
            out_dim_g=out_dim_g,
            norm_layer=norm_layer,
            res_connection=res_connection,
        )
        # Overwrite the nn to include sigmoid output
        self.sigmoid_output = sigmoid_output
        self.nn = SCOTCHFGNNI(
            self.stacked_key_masks,
            embedding_size,
            out_dim_g,
            norm_layer,
            res_connection,
            sigmoid_output=self.sigmoid_output,
        )


class SCOTCHFGNNI(FGNNI):
    """This adapts from FGNNI for SCOTCH by incorporting sigmoid output activation"""

    def __init__(
        self,
        group_mask: torch.Tensor,
        embedding_size: int | None = None,
        out_dim_g: int | None = None,
        norm_layer: type[nn.LayerNorm] | None = None,
        res_connection: bool = False,
        sigmoid_output: bool = False,
    ):
        super().__init__(
            group_mask=group_mask,
            embedding_size=embedding_size,
            out_dim_g=out_dim_g,
            norm_layer=norm_layer,
            res_connection=res_connection,
        )
        # Overwrite the f function to include sigmoid output
        out_dim_g = self.embedding_size if out_dim_g is None else out_dim_g
        in_dim_f = self.embedding_size + out_dim_g  # input dim of f
        a = max(4 * self.processed_dim_all, self.embedding_size, 64)
        self.f = generate_fully_connected(
            input_dim=in_dim_f,
            output_dim=self.processed_dim_all,
            hidden_dims=[a, a],
            non_linearity=nn.LeakyReLU,
            activation=nn.Identity if not sigmoid_output else nn.Sigmoid,
            device=group_mask.device,
            normalization=norm_layer,
            res_connection=res_connection,
        )
