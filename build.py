import torch
import gguf
from safetensors.torch import save_file

version = '1.0'
model_name = f'Maiasa_v{version}'


def main():

    TOKEN_TYPE_UNKNOWN = 2
    TOKEN_TYPE_CONTROL = 3
    TOKEN_TYPE_BYTE = 6

    byte_tokens = [f'<0x{i:02X}>' for i in range(256)]
    tokens = byte_tokens + ['<unk>', '<s>', '</s>']
    num_tokens = len(tokens)
    token_types = [TOKEN_TYPE_BYTE] * (num_tokens - 3) + [TOKEN_TYPE_UNKNOWN, TOKEN_TYPE_CONTROL, TOKEN_TYPE_CONTROL]

    bos_index = 257
    eos_index = 258
    a_index = 97  # 'a' = 97 in ascii

    dim = 16
    num_layers = 1
    num_heads = 1
    num_kv_heads = 1
    hidden_dim = 32

    token_emb = torch.zeros((num_tokens, dim), dtype=torch.float32)
    token_emb[:, 0] = 1.0
    token_emb[a_index, :] = 0.0
    token_emb[a_index, 1] = 1.0

    output_w = torch.full((num_tokens, dim), -100.0, dtype=torch.float32)
    output_w[a_index, :] = 0.0
    output_w[a_index, 0] = 10.0
    output_w[eos_index, :] = 0.0
    output_w[eos_index, 1] = 10.0

    state_dict = {
        'token_embd.weight': token_emb,
        'blk.0.attn_norm.weight': torch.ones(dim),
        'blk.0.ffn_norm.weight': torch.ones(dim),
        'output_norm.weight': torch.ones(dim),
        'blk.0.attn_q.weight': torch.zeros((dim, dim), dtype=torch.float32),
        'blk.0.attn_k.weight': torch.zeros((dim, dim), dtype=torch.float32),
        'blk.0.attn_v.weight': torch.zeros((dim, dim), dtype=torch.float32),
        'blk.0.attn_output.weight': torch.zeros((dim, dim), dtype=torch.float32),
        'blk.0.ffn_gate.weight': torch.zeros((hidden_dim, dim)),
        'blk.0.ffn_up.weight': torch.zeros((hidden_dim, dim)),
        'blk.0.ffn_down.weight': torch.zeros((dim, hidden_dim)),
        'output.weight': output_w,
    }

    save_file(state_dict, f'{model_name}.safetensors')

    gguf_writer = gguf.GGUFWriter(f'{model_name}.gguf', 'llama')

    gguf_writer.add_name(model_name)
    gguf_writer.add_architecture()
    gguf_writer.add_context_length(1024)
    gguf_writer.add_embedding_length(dim)
    gguf_writer.add_block_count(num_layers)
    gguf_writer.add_feed_forward_length(hidden_dim)
    gguf_writer.add_head_count(num_heads)
    gguf_writer.add_head_count_kv(num_kv_heads)
    gguf_writer.add_layer_norm_rms_eps(1e-5)
    gguf_writer.add_rope_dimension_count(dim // num_heads)
    gguf_writer.add_file_type(0)
    gguf_writer.add_tokenizer_model('llama')
    gguf_writer.add_token_list(tokens)
    gguf_writer.add_token_scores([0.0] * num_tokens)
    gguf_writer.add_token_types(token_types)
    gguf_writer.add_bos_token_id(bos_index)
    gguf_writer.add_eos_token_id(eos_index)
    gguf_writer.add_bool('tokenizer.ggml.byte_fallback', True)
    gguf_writer.add_tokenizer_pre('default')
    gguf_writer.add_token_merges([])

    for name, tensor in state_dict.items():
        gguf_writer.add_tensor(name, tensor.numpy())

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()


if __name__ == '__main__':
    main()
