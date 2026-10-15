from imuposer.models import *

def get_model(config=None, pretrained=None):
    model = config.model
    print(model)

    # load the dataset
    if model == "GlobalModelIMUPoser":
        net = IMUPoserModel(config=config)
    elif model == "ReconIMUPoser":
        net = ReconIMUPoserModel(config=config)
    elif model == "StagedIMUPoser":
        net = StagedIMUPoserModel(config=config)
    elif model == "TransformerIMUPoser":
        net = TransformerIMUPoser(config=config)
    elif model == "DiffusionIMUPoser":
        net = DiffusionIMUPoser(config=config)
    elif model == "AvatarPoserModel":
        net = AvatarPoserModel(config=config)
    elif model == "TokenIMUPoser":
        from imuposer.models.Transformers.TokenIMUPoser import TokenIMUPoser
        net = TokenIMUPoser(config=config)
    elif model in ("FlowIMUPoser", "DiffusionV2IMUPoser", "RegressIMUPoser"):
        from imuposer.models.Transformers.GenIMUPoser import FlowIMUPoser, DiffusionV2IMUPoser, RegressIMUPoser
        net = {"FlowIMUPoser": FlowIMUPoser, "DiffusionV2IMUPoser": DiffusionV2IMUPoser, "RegressIMUPoser": RegressIMUPoser}[model](config=config)
    elif model == "CNN1DIMUPoser":
        net = CNN1DIMUPoser(config=config)
    elif model == "CodebookIMUPoser":
        net = CodebookIMUPoser(config=config)
    elif model == "ActivityIMUPoser":
        net = ActivityIMUPoser(config=config)
    elif model == "GlobalModelIMUPoserFineTuneDIP":
        net = IMUPoserModelFineTune(config=config, pretrained_model=pretrained)
    else:
        print("Enter a valid model")
        return

    return net 
