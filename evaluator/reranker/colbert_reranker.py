import torch
from transformers import AutoModel, AutoTokenizer

from .base_reranker import BaseReranker


def insert_token(
    output: dict,
    insert_token_id: int,
    insert_position: int = 1,
    token_type_id: int = 0,
    attention_value: int = 1,
):
    """
    Inserts a new token at a specified position into the sequences of a tokenized representation.

    This function takes a dictionary containing tokenized representations
    (e.g., 'input_ids', 'token_type_ids', 'attention_mask') as PyTorch tensors,
    and inserts a specified token into each sequence at the given position.
    This can be used to add special tokens or other modifications to tokenized inputs.

    Parameters:
    - output (dict): A dictionary containing the tokenized representations. Expected keys
                     are 'input_ids', 'token_type_ids', and 'attention_mask'. Each key
                     is associated with a PyTorch tensor.
    - insert_token_id (int): The token ID to be inserted into each sequence.
    - insert_position (int, optional): The position in the sequence where the new token
                                       should be inserted. Defaults to 1, which typically
                                       follows a special starting token like '[CLS]' or '[BOS]'.
    - token_type_id (int, optional): The token type ID to assign to the inserted token.
                                     Defaults to 0.
    - attention_value (int, optional): The attention mask value to assign to the inserted token.
                                        Defaults to 1.

    Returns:
    - updated_output (dict): A dictionary containing the updated tokenized representations,
                             with the new token inserted at the specified position in each sequence.
                             The structure and keys of the output dictionary are the same as the input.
    """
    updated_output = {}
    for key in output:
        updated_tensor_list = []
        for seq in output[key]:
            first_part = seq[:insert_position]
            second_part = seq[insert_position:]
            new_element = (
                torch.tensor([insert_token_id])
                if key == "input_ids"
                else torch.tensor([token_type_id])
            )
            if key == "attention_mask":
                new_element = torch.tensor([attention_value])
            updated_seq = torch.cat((first_part, new_element, second_part), dim=0)
            updated_tensor_list.append(updated_seq)
        updated_output[key] = torch.stack(updated_tensor_list)
    return updated_output


class ColbertReranker(BaseReranker):
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        use_fp16=True,
        max_length=512,
        query_token: str = "[unused0]",
        document_token: str = "[unused1]",
    ):
        device = self._detect_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device
        self.model.to(device)
        if use_fp16 and "cuda" in device:
            self.model.half()
        self.model.eval()
        self.model.max_length = max_length
        self.max_length = max_length
        self.query_token_id: int = self.tokenizer.convert_tokens_to_ids(query_token)  # type: ignore
        self.document_token_id: int = self.tokenizer.convert_tokens_to_ids(document_token)  # type: ignore

    # base from: https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/postprocessor/llama-index-postprocessor-colbert-rerank/llama_index/postprocessor/colbert_rerank/base.py
    def _rerank(self, query: str, documents: list[str]) -> list[float]:
        query_encoding = self.tokenizer(
            query,
            return_tensors="pt",
            max_length=self.max_length - 1,  # for insert token
            truncation="longest_first",
        )
        query_encoding = insert_token(query_encoding, self.query_token_id)  # type: ignore
        query_encoding = {
            key: value.to(self.device) for key, value in query_encoding.items()
        }
        with torch.no_grad():
            query_embedding = self.model(**query_encoding).last_hidden_state
        rerank_score_list = []

        for document_text in documents:
            document_encoding = self.tokenizer(
                document_text,
                return_tensors="pt",
                truncation="longest_first",
                max_length=self.max_length - 1,  # for insert token
            )
            document_encoding = insert_token(document_encoding, self.document_token_id)  # type: ignore
            document_encoding = {
                key: value.to(self.device) for key, value in document_encoding.items()
            }
            with torch.no_grad():
                document_embedding = self.model(**document_encoding).last_hidden_state

            sim_matrix = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(2), document_embedding.unsqueeze(1), dim=-1
            )

            max_sim_scores, _ = torch.max(sim_matrix, dim=2)
            rerank_score_list.append(torch.mean(max_sim_scores, dim=1))
        sorted_scores = torch.stack(rerank_score_list).cpu().numpy()
        sorted_scores = sorted_scores.flatten().tolist()

        return sorted_scores
