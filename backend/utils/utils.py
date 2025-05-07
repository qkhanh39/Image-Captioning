import torch
import torchvision.transforms as transforms

def load_embedding_matrix():
    embedding_matrix = torch.load("assets/embedding_matrix.pt", map_location=torch.device("cpu"), weights_only=False)
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)
    return embedding_matrix

def load_i2w_and_w2i():
    w2i = torch.load("assets/w2i.pt", map_location=torch.device("cpu"), weights_only=False)
    i2w = torch.load("assets/i2w.pt", map_location=torch.device("cpu"), weights_only=False)

    w2i = {word: idx - 1 for word, idx in w2i.items()}
    i2w = {idx: word for word, idx in w2i.items()}
    return i2w, w2i

def load_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform

def generate_caption_beam_for_mlp(model, image, i2w, w2i, beam_width=3, max_len=33, device="cuda"):
    model.to(device)
    model.eval()

    with torch.no_grad():
        # Extract features
        image_tensor = image.to(device)
        features = model.vgg(image_tensor)
        if hasattr(model, 'kan'):
            features = model.kan(features)  # Apply KAN if available
        refined_features = model.feature_projector(features).unsqueeze(1)

        start_token = w2i["startseq"]
        end_token = w2i["endseq"]

        beam = [(0.0, [start_token], None)]  # (log_prob, token_list, hidden_state)

        for _ in range(max_len):
            new_beam = []

            for log_prob, tokens, hidden in beam:
                if tokens[-1] == end_token:
                    new_beam.append((log_prob, tokens, hidden))
                    continue

                input_tokens = torch.tensor([tokens[1:]], dtype=torch.long).to(device) if len(tokens) > 1 else torch.tensor([[start_token]], dtype=torch.long).to(device)
                caption_embed = model.embed(input_tokens)
                lstm_input = torch.cat((refined_features, caption_embed), dim=1)

                lstm_out, hidden_out = model.lstm(lstm_input, hidden)
                output = model.fc(lstm_out[:, -1, :])
                log_probs = torch.log_softmax(output, dim=1).squeeze(0)

                top_log_probs, top_indices = torch.topk(log_probs, beam_width)

                for i in range(beam_width):
                    next_token = top_indices[i].item()
                    total_log_prob = log_prob + top_log_probs[i].item()
                    new_tokens = tokens + [next_token]
                    new_beam.append((total_log_prob, new_tokens, hidden_out))

            # Keep top k beams
            beam = sorted(new_beam, key=lambda x: x[0], reverse=True)[:beam_width]

        # Retrieve the best caption
        best_tokens = beam[0][1]
        caption_words = [i2w[idx] for idx in best_tokens[1:] if i2w[idx] != "endseq"]

        return " ".join(caption_words)