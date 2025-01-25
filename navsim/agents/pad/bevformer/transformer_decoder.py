import torch
import torch.nn as nn

class MyTransformeDecoder(nn.Module):
    def __init__(self, config,input_dim,output_dim,trajenc=True,add_embeding=0,projout=True,layer_num=2):
        super().__init__()

        d_ffn = config.tf_d_ffn
        d_model = config.tf_d_model

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=config.tf_num_head,
            dim_feedforward=d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        self.trajenc=trajenc

        if self.trajenc:
            self.in_proj = nn.Linear(input_dim, d_model)
        else:
            self.init_feature= nn.Embedding(input_dim, d_model)

        self.add_embeding=add_embeding

        if add_embeding!=0:
            self.add_embed = nn.Embedding(add_embeding, d_model)

        self.decoder = nn.TransformerDecoder(decoder_layer, layer_num)
       #self.decoder = _get_clones(decoder_layer, layer_num)

        self.projout=projout
        if self.projout:
            self.out_proj = nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, output_dim),
            )


    def forward(self,query,keyval,pos_embed=None,tgt_mask=None):

        if query is not None:
            query_feature=self.in_proj(query)
        else:
            query_feature=self.init_feature.weight[None].repeat(keyval.shape[0],1, 1)

        if self.add_embeding:
            query_feature=query_feature[:,None]+ self.add_embed.weight[None,:,None]

        query_feature=query_feature.reshape(query_feature.shape[0], -1, query_feature.shape[-1])

        if pos_embed is not None:
            query_feature=query_feature+pos_embed

        # for layer in self.decoder:
        #     query_feature, keyval = layer(
        #         query_feature, None, keyval, None,
        #     )
        # for mod in self.decoder:
        #     output = mod(query_feature,keyval, tgt_is_causal=tgt_is_causal)


        output=self.decoder(query_feature,keyval,tgt_mask=tgt_mask)
        if self.projout:
            output=self.out_proj(output)

        return output



