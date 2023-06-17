# %%
import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
# %%
from neel.imports import *
from neel_plotly import *

# %%
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_grad_enabled(False)
# %%
model = HookedTransformer.from_pretrained("pythia-160m")
# %%
color_probes = pickle.load(open("/workspace/ProbingRGB/color_probes (1).p", "rb"))
# print(color_probes.shape)
list(color_probes.keys())
# %%
def gen_hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"
num_prompts = 512
np.random.seed(SEED)
rgb = np.random.randint(0, 256, size=(num_prompts, 3))
hexes = [gen_hex(r, g, b) for r, g, b in rgb]
print(hexes[1], rgb[1])
# %%
for i in range(5):
    print(hexes[i], model.to_str_tokens(hexes[i]))
# %%
# Stupid example: e8a0a7 -> 6 tokens
hex_tokens = model.to_tokens(hexes)
print(hex_tokens.shape)
final_index = (hex_tokens!=model.tokenizer.pad_token_id).sum(dim=-1)
filter = ~((final_index ==3) | (final_index==7))
final_index = final_index[filter]
hex_tokens = hex_tokens[filter]
hexes = np.array(hexes)[filter.tolist()]
rgb = rgb[filter.tolist()]
print(final_index[0], hex_tokens[0], hex_tokens[0, :final_index[0]+1], hex_tokens[0, final_index[0]])
num_prompts = filter.sum().item()
print(num_prompts)
print(rgb.shape, hexes.shape, hex_tokens.shape, final_index.shape)
# %%
logits, cache = model.run_with_cache(hex_tokens)
stacked_resid = cache.stack_activation("resid_post")
print(f"{stacked_resid.shape=}") # layer batch pos d_model
final_resid = stacked_resid[:, torch.arange(num_prompts).cuda(), final_index, :]
print(final_resid.shape)
print(final_resid[0, 0, 0])
print(stacked_resid[0, 0, :, 0])
print(hex_tokens[0])
# %%
layer = 3
l3_red_probe = torch.tensor(color_probes[("pythia-160m", "hex_color", str(layer), "red")]).cuda()
red_probe_vals = final_resid[layer] @ l3_red_probe
red_true_vals = torch.tensor(rgb[:, 0]).cuda()
px.scatter(x=red_probe_vals.tolist(), y=red_true_vals.float().tolist(), labels={"x":"Probe", "y":"True"}, title="Red value of Probe vs True value after layer 3", trendline="ols", color=final_index.tolist(), hover_name=np.array(hexes)).show()
# scatter(x=red_probe_vals.tolist(), y=red_true_vals.float().tolist(), xaxis="Probe", yaxis="True", title="Red of Probe vs True value", trendline="ols")
# %%
# Direct Probe Attribution
probe_dir = l3_red_probe / l3_red_probe.norm()
full_decomp_resid, resid_labels = cache.get_full_resid_decomposition(layer+1, expand_neurons=False, return_labels=True)
decomp_resid = full_decomp_resid[:, torch.arange(num_prompts).cuda(), final_index, :]
print(decomp_resid.shape)
decomp_resid_probe = decomp_resid @ probe_dir

line(decomp_resid_probe.mean(dim=-1), x=resid_labels, title="Mean Contribution of components to probe")
line(decomp_resid_probe.std(dim=-1), x=resid_labels, title="Stdev Contribution of components to probe")

# %%
is_top_half = torch.tensor(rgb[:, 0] > 128).cuda()
line([decomp_resid_probe[:, is_top_half].mean(dim=-1), decomp_resid_probe[:, ~is_top_half].mean(dim=-1), decomp_resid_probe[:, is_top_half].mean(dim=-1) - decomp_resid_probe[:, ~is_top_half].mean(dim=-1)], x=resid_labels, title="Mean Contribution of components to probe by top vs bottom half", line_labels=["top_half", "bottom_half", "diff"])
# %%
component_label = ["L2H8", "L3H4", "L3H8", "3_mlp_out"]
for comp in component_label:
    i = resid_labels.index(comp)
    px.scatter(x=red_probe_vals.tolist(), y=decomp_resid_probe[i].tolist(), labels={"x":"Probe", "y":comp}, title=f"Red value of full Probe vs {comp} contribution value after layer 3", trendline="ols", color=final_index.tolist(), hover_name=np.array(hexes)).show()
    px.scatter(x=red_true_vals.tolist(), y=decomp_resid_probe[i].tolist(), labels={"x":"True", "y":comp}, title=f"Red true value vs {comp} contribution value after layer 3", trendline="ols", color=final_index.tolist(), hover_name=np.array(hexes)).show()
# %%
layer = 3
head = 4
label = f"L{layer}H{head}"

for i in range(10):
    imshow(cache["pattern", layer][:, head][i], title=f"Pattern for {label} for {hexes[i]}", x=nutils.process_tokens_index(hex_tokens[i], model), y=nutils.process_tokens_index(hex_tokens[i], model))
# %%
first_token_attn = []
second_token_attn = []
for i in range(num_prompts):
    first_token_attn.append(cache["attn", layer][i, head, final_index[i], 2])
    second_token_attn.append(cache["attn", layer][i, head, final_index[i], 3])
first_token_attn = torch.tensor(first_token_attn)
second_token_attn = torch.tensor(second_token_attn)
histogram(torch.stack([first_token_attn, second_token_attn], dim=-1), color=["first", "second"], title=f"Attention to first and second token for {label}", barmode="overlay")

first_token_lengths = torch.tensor([len(i) for i in model.to_str_tokens(hex_tokens[:, 2])])

prompt_df = pd.DataFrame({
    "hex": hexes,
    "first_token_attn": first_token_attn.tolist(),
    "second_token_attn": second_token_attn.tolist(),
    "first_token_length": first_token_lengths.tolist(),
})
prompt_df["attn_rest"] = 1 - prompt_df["first_token_attn"] - prompt_df["second_token_attn"]
px.histogram(prompt_df, x="first_token_attn", color="first_token_length", title=f"Attention to first token for {label}", barmode="overlay").show()
px.histogram(prompt_df, x="second_token_attn", color="first_token_length", title=f"Attention to second token for {label}", barmode="overlay").show()
px.histogram(prompt_df, x="attn_rest", color="first_token_length", title=f"Attention to non-first-two token for {label}", barmode="overlay").show()
# %%
prompt_df["is_top_half"] = is_top_half.tolist()
red_attn = []
for row in prompt_df.itertuples():
    if row.first_token_length==1:
        red_attn.append(row.second_token_attn)
    else:
        red_attn.append(row.first_token_attn)
prompt_df["red_attn"] = red_attn
prompt_df["red"] = rgb[:, 0].tolist()
prompt_df["probe_attr"] = decomp_resid_probe[resid_labels.index(label)].tolist()
prompt_df.head()
# %%
px.box(prompt_df, x="first_token_length", y="red_attn", color="is_top_half", title=f"Attention to red token for {label} by top vs bottom half", points="all").show()

# %%
W_O = model.W_O[layer, head]
W_probe = W_O @ probe_dir
value_attr = cache["v", layer][:, 2:4, head, :] @ W_probe
first_tok_attns = cache["pattern", layer][torch.arange(num_prompts).cuda(), head, final_index, 2:4]
is_single_char = torch.tensor((prompt_df.first_token_length==1).values)
line(value_attr.T, title="Value Attrs")
line(first_tok_attns.T, title="First Token Attns")
# %%
prompt_df["first_value_attr"] = value_attr[:, 0].tolist()
prompt_df["second_value_attr"] = value_attr[:, 1].tolist()
prompt_df["first_value_weighted_attr"] = prompt_df["first_value_attr"] * prompt_df["first_token_attn"]
prompt_df["second_value_weighted_attr"] = prompt_df["second_value_attr"] * prompt_df["second_token_attn"]
prompt_df["value_weighted_attr"] = np.where(prompt_df["first_token_length"]==1, prompt_df["second_value_weighted_attr"], prompt_df["first_value_weighted_attr"])
prompt_df["value_attr"] = np.where(prompt_df["first_token_length"]==1, prompt_df["second_value_attr"], prompt_df["first_value_attr"])

prompt_df.head()
# %%
px.scatter(prompt_df, x="value_attr", y="red", color="first_token_length", title=f"Value attr vs red value for {label}", trendline="ols", hover_name="hex").show()
px.scatter(prompt_df, x="value_weighted_attr", y="red", color="first_token_length", title=f"Weighted Value attr vs red value for {label}", trendline="ols", hover_name="hex").show()
px.scatter(prompt_df, x="value_attr", y="red", color="first_token_length", title=f"Value attr vs red value for {label}", trendline="ols", hover_name="hex").show()

# %%
px.scatter(prompt_df, x="first_value_attr", y="red", color="first_token_length", title=f"First Value attr vs red value for {label}", trendline="ols", hover_name="hex").show()
px.scatter(prompt_df, x="second_value_attr", y="red", color="first_token_length", title=f"Second Value attr vs red value for {label}", trendline="ols", hover_name="hex").show()

# %%
probe_OV = model.W_V[layer, head] @ model.W_O[layer, head] @ probe_dir

full_decomp_resid_head, resid_head_labels = cache.get_full_resid_decomposition(3, expand_neurons=False, apply_ln=True, pos_slice=2, return_labels=True)
value_attr_probe = (full_decomp_resid_head[:, ~is_single_char, :] @ probe_OV)

is_top_half_given_multi = is_top_half[~is_single_char]

line([value_attr_probe[:, is_top_half_given_multi].mean(-1), value_attr_probe[:, ~is_top_half_given_multi].mean(-1), value_attr_probe[:, is_top_half_given_multi].mean(-1) - value_attr_probe[:, ~is_top_half_given_multi].mean(-1)], line_labels=["top", "bottom", "diff"], title="Value Attribution by top vs bottom half", x=resid_head_labels)
# %%
scatter(x=value_attr_probe[-3, :], y=torch.tensor(rgb[:, 0])[~is_single_char], title="Value Attr vs red value for all multi-token red chars (MLP2)", xaxis="Value Attr", yaxis="Red Value", hover=hexes[~to_numpy(is_single_char)])
scatter(x=value_attr_probe[-5, :], y=torch.tensor(rgb[:, 0])[~is_single_char], title="Value Attr vs red value for all multi-token red chars (MLP0)", xaxis="Value Attr", yaxis="Red Value", hover=hexes[~to_numpy(is_single_char)])
# %%
begins_with_letter = torch.tensor([not hexes[i][2].isalpha() for i in range(num_prompts)]).cuda()
print(begins_with_letter)

filter_1 = begins_with_letter & (~is_single_char.cuda())
filter_2 = (~begins_with_letter) & (~is_single_char.cuda())
mlp_acts_with_letter = cache["post", 2][filter_1, 2, :]
mlp_acts_without_letter = cache["post", 2][filter_2, 2, :]
line([mlp_acts_with_letter.mean(0), mlp_acts_without_letter.mean(0), mlp_acts_with_letter.mean(0)-mlp_acts_without_letter.mean(0)], line_labels=["letter", "number", "diff"], title="MLP2 activations by first token type", xaxis="Neuron", yaxis="Ave Act")
# %%
neuron_df = pd.DataFrame({
    "neuron": np.arange(model.cfg.d_mlp),
    "letter_act": mlp_acts_with_letter.mean(0).tolist(),
    "number_act": mlp_acts_without_letter.mean(0).tolist(),
    "diff": (mlp_acts_with_letter.mean(0) - mlp_acts_without_letter.mean(0)).tolist()
})

neuron_weights = model.W_out[2] @ probe_OV
print(neuron_weights.shape)
neuron_df["weight"] = neuron_weights.tolist()
neuron_df["abs_weight"] = neuron_df["weight"].abs()
neuron_df["letter_weight"] = neuron_df["weight"] * neuron_df["letter_act"]
neuron_df["number_weight"] = neuron_df["weight"] * neuron_df["number_act"]
neuron_df["diff_weight"] = neuron_df["weight"] * neuron_df["diff"]
px.line(neuron_df, y=["letter_weight", "number_weight", "diff_weight"], title="Neuron effect on probe from MLP2 to probe via L3H4 OV")
# %%
# Linear Regression

# %%
# BOS Ablate each head and look at loss damage