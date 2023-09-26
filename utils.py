import torch


specials = {'bert': '#', 'gpt2': 'Ġ', 'xlnet': '▁', 'roberta': 'Ġ', 'indic': '▁'}

def select_indices(tokens, raw_tokens, special, mode):
    mask = []
    raw_i = 0
    collapsed = ''

    for i in range(len(tokens)):
        token = tokens[i]

        while len(token) > 0 and token[0] == special:
            token = token[1:]
        if collapsed == '' and len(token) > 0:
            start_idx = i
        collapsed += token
        if collapsed == raw_tokens[raw_i]:
            if mode == 'first':
                mask.append(start_idx)
            elif mode == 'last':
                mask.append(i)
            else:
                raise NotImplementedError
            raw_i += 1
            collapsed = ''
    if raw_i != len(raw_tokens):
        raise Exception(f'Token mismatch: \n{tokens}\n{raw_tokens}')
    return mask


def group_indices(tokens, raw_tokens, special):
    mask = []
    raw_i = 0

    collapsed = ''
    options = [raw_tokens[raw_i]]
    skip = 0
    collapsed_cnt = 0
    for i in range(len(tokens)):
        token = tokens[i]

        while len(token) > 0 and token[0] == special:
            token = token[1:]

        collapsed_cnt += 1
        if token != '[UNK]':
            collapsed += token
            if collapsed in options:
                raw_tokens_cnt = options.index(collapsed)
                for j in range(raw_tokens_cnt+1):
                    mask.append(raw_i)
                    raw_i += 1
                for j in range(collapsed_cnt-raw_tokens_cnt-1):
                    mask.append(raw_i-1)
                if raw_i >= len(raw_tokens):
                    if i != len(tokens)-1:
                        raise Exception("Tokens more that tags.")
                    break
                options = [raw_tokens[raw_i]]
                collapsed = ''
                collapsed_cnt = 0
                skip = 0
        else:
            if collapsed:
                print(options, collapsed)
                raise Exception("Invalid token-tags!")
            skip += 1
            options.append(raw_tokens[raw_i+skip])

    if raw_i != len(raw_tokens):
        print(options, collapsed)
        return 
    return torch.tensor(mask)