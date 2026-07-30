from transformers import RobertaForMaskedLM


def create_model(config):
    return RobertaForMaskedLM(config.create_model_config())
