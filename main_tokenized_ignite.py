import os
import time

import torch
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, EarlyStopping, create_lr_scheduler_with_warmup, CosineAnnealingScheduler
from ignite.handlers import Timer
from ignite.metrics import Accuracy
from torch.nn import TripletMarginLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from transformers.modeling_outputs import MaskedLMOutput

import my_setup
from data_processing.masked_sequence_model_token_based_triplet_dataset import Sequences, SequenceToSequenceDataset
from loss_function.cross_entropy_with_class_ignorance_loss import masked_cross_entropy_loss
from models.simple_model import TransformerModel2, TokenLevelClassifier, SimpEncoderDecoderModel
from utils.my_utils import short_summary


device_training = torch.device(my_setup.DEVICE_TRAINING)
print(f"device = {device_training}")
print(f"device_training = {device_training}")


def train_step(engine, batch):
    global model, optimizer, masked_criterion
    model.train()
    optimizer.zero_grad()
    masked_tokens, tokens, masks = batch
    masked_tokens = masked_tokens.to(device_training)
    tokens = tokens.to(device_training)
    masks = masks.to(device_training)
    # pred_logits = model(masked_tokens, masks)
    pred_logits = model(masked_tokens, masks=None)
    loss = masked_criterion(pred_logits, tokens, masks)
    loss.backward()
    optimizer.step()
    return loss.item()


def eval_step(engine, batch):
    global model
    model.eval()
    with torch.no_grad():
        masked_tokens, tokens, masks = batch
        masked_tokens = masked_tokens.to(device_training)
        tokens = tokens.to(device_training)
        masks = masks.to(device_training)
        pred_logits = model(masked_tokens, masks)
        loss = masked_criterion(pred_logits, tokens, masks)
        # pred_logits = pred_logits.transpose(-1, -2)
        # pred_dists = torch.softmax(pred_logits, dim=2)
        # pred_tokens = torch.argmax(pred_dists, dim=2)
        return {'y_pred': pred_logits,
                'y_true': tokens,
                'masks': masks,
                'loss': loss.item()
                }


def train_step_triplets(engine, batch):
    global model, optimizer, masked_criterion, triplet_criterion
    model.train()
    optimizer.zero_grad()
    prepared_positive, prepared_negative = batch
    masked_tokens_positive, tokens_positive, masks_positive = prepared_positive
    masked_tokens_negative, tokens_negative, masks_negative = prepared_negative

    masked_tokens_positive = masked_tokens_positive.to(device_training)
    masked_tokens_negative = masked_tokens_negative.to(device_training)

    tokens_positive = tokens_positive.to(device_training)
    tokens_negative = tokens_negative.to(device_training)

    masks_positive = masks_positive.to(device_training)
    masks_negative = masks_negative.to(device_training)

    anchor = tokens_positive

    # The single passing approach takes extremly long in the backward pass
    # (however, the multiple calls to model are not the issue. Instead the Batch Size is Adapted to 128 as this yields a 5 fold speedup in with halv the data in comparison to 256)
    pred_logits_anchor, encoding_anchor = model(anchor, output_encoding=True)
    # pred_logits_positive, encoding_positive = model(masked_tokens_positive, masks_positive, output_encoding=True)
    # pred_logits_negative, encoding_negative = model(masked_tokens_negative, masks_negative, output_encoding=True)
    pred_logits_positive, encoding_positive = model(masked_tokens_positive, masks=None, output_encoding=True)
    pred_logits_negative, encoding_negative = model(masked_tokens_negative, masks=None, output_encoding=True)

    # masks_anchor = torch.zeros(anchor.size())
    #
    # token_input = torch.cat((anchor, masked_tokens_positive, masked_tokens_negative), dim=0)
    # masks_input = torch.cat((masks_anchor, masks_positive, masks_negative), dim=0)
    # #token_input = masked_tokens_positive
    # #masks_input = masks_positive
    #
    # pred_logits, encoding = model(token_input, masks_input, output_encoding=True)
    # pred_logits_anchor, pred_logits_positive, pred_logits_negative = torch.split(pred_logits, len(anchor), dim=0)
    # encoding_anchor, encoding_positive, encoding_negative = torch.split(encoding, len(anchor), dim=0)
    ## pred_logits_positive, encoding_positive = pred_logits, encoding
    anchor_embedding = encoding_anchor[:, 0]
    positive_embedding = encoding_positive[:, 0]
    negative_embedding = encoding_negative[:, 0]

    loss_positive = masked_criterion(pred_logits_positive, tokens_positive, masks_positive)
    loss_negative = masked_criterion(pred_logits_negative, tokens_negative, masks_negative)
    loss_triplet = triplet_criterion(anchor_embedding, positive_embedding, negative_embedding)
    loss = loss_positive + loss_negative + loss_triplet

    # loss = loss_triplet
    loss.backward()
    optimizer.step()
    return loss.item()


def eval_step_triplets(engine, batch):
    global model
    model.eval()
    with torch.no_grad():
        prepared_positive, prepared_negative = batch
        masked_tokens_positive, tokens_positive, masks_positive = prepared_positive
        masked_tokens_negative, tokens_negative, masks_negative = prepared_negative

        masked_tokens_positive = masked_tokens_positive.to(device_training)
        masked_tokens_negative = masked_tokens_negative.to(device_training)

        tokens_positive = tokens_positive.to(device_training)
        tokens_negative = tokens_negative.to(device_training)

        masks_positive = masks_positive.to(device_training)
        masks_negative = masks_negative.to(device_training)

        anchor = tokens_positive

        # The single passing approach takes extremly long in the backward pass
        pred_logits_anchor, encoding_anchor = model(anchor, output_encoding=True)
        pred_logits_positive, encoding_positive = model(masked_tokens_positive, masks_positive, output_encoding=True)
        pred_logits_negative, encoding_negative = model(masked_tokens_negative, masks_negative, output_encoding=True)

        # masks_anchor = torch.zeros(anchor.size())
        #
        # token_input = torch.cat((anchor, masked_tokens_positive, masked_tokens_negative), dim=0)
        # masks_input = torch.cat((masks_anchor, masks_positive, masks_negative), dim=0)
        #
        # pred_logits, encoding = model(token_input, masks_input, output_encoding=True)
        #
        # pred_logits_anchor, pred_logits_positive, pred_logits_negative = torch.split(pred_logits, len(anchor), dim=0)
        # encoding_anchor, encoding_positive, encoding_negative = torch.split(encoding, len(anchor), dim=0)

        anchor_embedding = encoding_anchor[:, 0]
        positive_embedding = encoding_positive[:, 0]
        negative_embedding = encoding_negative[:, 0]

        loss_positive = masked_criterion(pred_logits_positive, tokens_positive, masks_positive)
        loss_negative = masked_criterion(pred_logits_negative, tokens_negative, masks_negative)
        loss_triplets = triplet_criterion(anchor_embedding, positive_embedding, negative_embedding)
        loss = loss_positive + loss_negative + loss_triplets

        pred_logits = torch.concatenate((pred_logits_positive, pred_logits_negative), dim=0)
        tokens = torch.concatenate((tokens_positive, tokens_negative), dim=0)
        masks = torch.concatenate((masks_positive, masks_negative), dim=0)
        return {'y_pred': pred_logits,
                'y_true': tokens,
                'masks': masks,
                'loss': loss.item(),
                'losses': {'loss_positive': loss_positive,
                           'loss_negative': loss_negative,
                           'loss_triplets': loss_triplets}
                }


def train_step_triplets_classic(engine, batch):
    global model, optimizer, masked_criterion
    # print("Training Step")

    masked_tokens, tokens, masks = batch
    masked_tokens = masked_tokens.to(device_training)
    tokens = tokens.to(device_training)
    masks = masks.to(device_training)

    """
    Selection of Hard Negatives
    """
    model.eval()

    logits, encodings = model(tokens, output_encoding=True)
    embeddings = encodings[:, 0]
    expanded_embeddings = embeddings.unsqueeze(1)
    dist_mat = torch.norm(expanded_embeddings - embeddings, p=2, dim=-1)
    # Prevents the sequence from being it's own negative
    dist_mat += torch.diag((torch.ones(len(embeddings)) * torch.inf).to(device_training))
    hard_negatives_ids = torch.argmin(dist_mat, -1)

    """
    Setup of Training
    """
    model.train()
    optimizer.zero_grad()
    model.train()
    optimizer.zero_grad()
    masked_tokens_positive = masked_tokens
    tokens_positive = tokens
    masks_positive = masks
    masked_tokens_negative = masked_tokens[hard_negatives_ids]
    tokens_negative = tokens_positive[hard_negatives_ids]
    masks_negative = masks[hard_negatives_ids]

    anchor = tokens_positive

    pred_logits_anchor, encoding_anchor = model(anchor, masks=None, output_encoding=True)
    pred_logits_positive, encoding_positive = model(masked_tokens_positive, masks=None, output_encoding=True)
    pred_logits_negative, encoding_negative = model(masked_tokens_negative, masks=None, output_encoding=True)

    anchor_embedding = encoding_anchor[:, 0]
    positive_embedding = encoding_positive[:, 0]
    negative_embedding = encoding_negative[:, 0]

    loss_positive = masked_criterion(pred_logits_positive, tokens_positive, masks_positive)
    loss_negative = masked_criterion(pred_logits_negative, tokens_negative, masks_negative)
    loss_triplet = triplet_criterion(anchor_embedding, positive_embedding, negative_embedding)
    loss = loss_positive + loss_negative + loss_triplet

    # loss = loss_triplet
    loss.backward()
    optimizer.step()
    return loss.item()


def eval_step_triplets_classic(engine, batch):
    global model
    model.eval()
    # print("Eval Step")
    with torch.no_grad():
        masked_tokens, tokens, masks = batch
        masked_tokens = masked_tokens.to(device_training)
        tokens = tokens.to(device_training)
        masks = masks.to(device_training)

        """
        Selection of Hard Negatives
        """
        model.eval()

        logits, encodings = model(tokens, output_encoding=True)
        embeddings = encodings[:, 0]
        expanded_embeddings = embeddings.unsqueeze(1)
        dist_mat = torch.norm(expanded_embeddings - embeddings, p=2, dim=-1)
        # Prevents the sequence from being it's own negative
        dist_mat += torch.diag((torch.ones(len(embeddings)) * torch.inf).to(device_training))
        hard_negatives_ids = torch.argmin(dist_mat, -1)

        """
        Setup of Evaluation
        """
        model.train()
        optimizer.zero_grad()
        masked_tokens_positive = masked_tokens
        tokens_positive = tokens
        masks_positive = masks
        masked_tokens_negative = masked_tokens[hard_negatives_ids]
        tokens_negative = tokens_positive[hard_negatives_ids]
        masks_negative = masks[hard_negatives_ids]

        anchor = tokens_positive

        pred_logits_anchor, encoding_anchor = model(anchor, output_encoding=True)
        pred_logits_positive, encoding_positive = model(masked_tokens_positive, masks_positive, output_encoding=True)
        pred_logits_negative, encoding_negative = model(masked_tokens_negative, masks_negative, output_encoding=True)

        anchor_embedding = encoding_anchor[:, 0]
        positive_embedding = encoding_positive[:, 0]
        negative_embedding = encoding_negative[:, 0]

        loss_positive = masked_criterion(pred_logits_positive, tokens_positive, masks_positive)
        loss_negative = masked_criterion(pred_logits_negative, tokens_negative, masks_negative)
        loss_triplets = triplet_criterion(anchor_embedding, positive_embedding, negative_embedding)
        loss = loss_positive + loss_negative + loss_triplets

        pred_logits = torch.concatenate((pred_logits_positive, pred_logits_negative), dim=0)
        tokens = torch.concatenate((tokens_positive, tokens_negative), dim=0)
        masks = torch.concatenate((masks_positive, masks_negative), dim=0)
        return {'y_pred': pred_logits,
                'y_true': tokens,
                'masks': masks,
                'loss': loss.item(),
                'losses': {'loss_positive': loss_positive,
                           'loss_negative': loss_negative,
                           'loss_triplets': loss_triplets}
                }


if __name__ == '__main__':
    # For TOKENIZER = DNABertTokenizer() and the Randomised Sequences File, the Loss needs to get below one to be equivalently good as the pretrained DNA Bert
    """
    Get Data
    """

    num_epochs = 150

    # file_path = my_setup.RANDOMISED_SEQUENCES_FILE
    file_path = my_setup.NGS_SEQUENCES_FILE
    batch_size = 128  # 1024
    if my_setup.TRIPLET_MODE:
        batch_size = 64  # Batch size 64 or 128 is a good trade of between batch size and training speed per sample .
    n_repetitions = 1
    split = [0.7, 0.15, 0.15]  # Currently without effect es test and training are pre partioned


    limit = None

    # Initial Learning Rate for Learning Rate Schedule
    learning_rate = 8 * 10 ** (-4)  # Used for currently best run

    tokenizer = my_setup.TOKENIZER

    train_sequences = Sequences(file_path.replace(".txt", "_train.txt"), limit=limit)
    val_sequences = Sequences(file_path.replace(".txt", "_val.txt"), limit=limit)
    test_sequences = Sequences(file_path.replace(".txt", "_test.txt"), limit=limit)

    sequences = [train_sequences, val_sequences, test_sequences]
    sequences_split = [seqs.data for seqs in sequences]

    datasets = [SequenceToSequenceDataset(data, n_repetitions=n_repetitions, tokenizer=tokenizer)
                for data in sequences_split]
    pin_memory = True if my_setup.DEVICE == "cpu" else False
    data_loaders = [DataLoader(dataset=dataset, batch_size=batch_size,
                               shuffle=True, pin_memory=pin_memory, num_workers=0)
                    for dataset in datasets]
    train_loader, validation_loader, test_loader = data_loaders

    print("Loaded Data")

    """
    Create Model
    """

    num_base_tokens = len(tokenizer.BASE_TOKENS)
    vocab_size = tokenizer.VOCABULARY_SIZE

    embedding_dim = my_setup.EMBEDDING_DIM
    encoder_model = TransformerModel2(max_sequence_length=my_setup.MAX_SEQUENCE_LENGTH,
                                      vocabulary_size=tokenizer.VOCABULARY_SIZE,
                                      embedding_dim=embedding_dim,
                                      dropout=0.1)

    token_level_classifier = TokenLevelClassifier(input_dim=embedding_dim, output_dim=vocab_size)


    encoder_decoder_model = SimpEncoderDecoderModel(encoder=encoder_model, decoder=token_level_classifier)

    load_parameters = False
    if load_parameters:
        encoder_decoder_parameters_path = my_setup.MODEL_PARAMETERS_DIRECTORY + "encoder_decoder_parameters.pth"
        encoder_decoder_parameters_path = "model_parameters_93%_Validation_accuracy/model_dim=128_max-len=168_triplet=False_best_model_val_loss=0.9332.pt"
        if not os.path.exists(encoder_decoder_parameters_path):
            raise Exception(
                f"The required parameters of the pretrained encoded do not exist at {encoder_decoder_parameters_path}.")

        state_dict = torch.load(encoder_decoder_parameters_path, map_location=torch.device('cpu'))
        encoder_decoder_model.load_state_dict(state_dict)
        print("Loaded Parameters")

    model = encoder_decoder_model
    print("Created Model")

    short_summary(model)

    model = model.to(device_training)
    print("Moved model to:", device_training)

    """
    Setup training
    """
    parameters = model.parameters()
    optimizer = Adam(params=parameters, lr=learning_rate, weight_decay=10 ** (-7))
    masked_criterion = masked_cross_entropy_loss

    triplet_criterion = TripletMarginLoss(margin=1.0, reduction="mean")

    trainer = Engine(train_step)
    evaluator = Engine(eval_step)
    if my_setup.TRIPLET_MODE:
        if my_setup.TRIPLET_MODE_CLASSIC_NEGATIVE_SELECTION:
            trainer = Engine(train_step_triplets_classic)
            evaluator = Engine(eval_step_triplets_classic)
            print("Loaded Classic Triplet Selection Training Steps")
        else:
            trainer = Engine(train_step_triplets)
            evaluator = Engine(eval_step_triplets)
            print("Loaded Dataset wide custom Triplet Selection Training Steps")

    if True:
        torch_lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=0.98, verbose=True)

        warmup_epochs = 4
        scheduler = create_lr_scheduler_with_warmup(torch_lr_scheduler,
                                                    warmup_start_value=10 ** (-6),
                                                    warmup_end_value=learning_rate,
                                                    warmup_duration=warmup_epochs)


        trainer.add_event_handler(Events.EPOCH_STARTED, scheduler)

    handler = Timer(average=False)
    handler.attach(trainer)
    handler.attach(evaluator)


    # Only returns the predictions and labels of the masked prediction task and not of the whole sequence.
    def accuracy_transform(results):
        masks = results["masks"]
        y_logits = results["y_pred"]
        y = results["y_true"]
        y_pred = y_logits[masks]
        y_true = y[masks]
        return y_pred, y_true


    metrics = {
        "accuracy": Accuracy(output_transform=accuracy_transform),
        # "loss": Loss(masked_criterion)
    }
    if my_setup.TRIPLET_MODE:
        metrics = {
            "accuracy": Accuracy(output_transform=accuracy_transform),
        }

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    log_interval = 100


    # Ignite Tutorial for Model Checkpoint and Early Stopping
    # https://www.kaggle.com/code/vfdev5/pytorch-and-ignite-on-fruits-360

    def score_function(engine):
        output = engine.state.output
        accuracy = engine.state.metrics['accuracy']
        triplet_loss = float(output["losses"]["loss_triplets"]) if "losses" in output else 0
        loss = engine.state.output['loss']
        if my_setup.TRIPLET_MODE:
            score = accuracy - triplet_loss
        else:
            score = accuracy
        # Early stopping and model checkpoint focus on highest scores
        print("Score:", score)
        return score


    setup_id = f"dim={my_setup.EMBEDDING_DIM}_max-len={my_setup.MAX_SEQUENCE_LENGTH}_triplet={my_setup.TRIPLET_MODE}"
    best_model_saver = ModelCheckpoint("_best_models",  # folder where to save the best model(s)
                                       filename_prefix=f"model_{setup_id}",
                                       # filename prefix -> {filename_prefix}_{name}_{step_number}_{score_name}={abs(score_function_result)}.pth
                                       score_name="val_loss",
                                       score_function=score_function,
                                       n_saved=3,
                                       atomic=True,
                                       # objects are saved to a temporary file and then moved to final destination, so that files are guaranteed to not be damaged
                                       save_as_state_dict=True,  # Save object as state_dict
                                       create_dir=True,
                                       require_empty=False)

    evaluator.add_event_handler(Events.COMPLETED, best_model_saver, {"best_model": model})

    training_saver = ModelCheckpoint("_checkpoint",
                                     filename_prefix=f"checkpoint_{setup_id}",
                                     # save_interval=1000,
                                     n_saved=1,
                                     atomic=True,
                                     save_as_state_dict=True,
                                     create_dir=True,
                                     require_empty=False)

    to_save = {"model": model, "optimizer": optimizer, "lr_scheduler": scheduler}
    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=1000), training_saver, to_save)

    early_stopping = EarlyStopping(patience=20, score_function=score_function, trainer=trainer)
    # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
    evaluator.add_event_handler(Events.COMPLETED, early_stopping)


    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        print(
            f"Epoch [{engine.state.epoch}], Iter [{engine.state.iteration}] - Loss: {engine.state.output:.4f} (lr = {optimizer.param_groups[0]['lr']})")


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_loss(engine):
        print(f"Epoch {engine.state.epoch} - Loss: {engine.state.output:.4f} (duration {handler.value()} s)")


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_loss(engine):
        evaluator.run(validation_loader)
        metrics = evaluator.state.metrics
        output = evaluator.state.output

        epoch_info = f"Epoch {engine.state.epoch}"
        validation_info = f"Validation Loss: {output['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}"
        metric_info = ""
        for metric in metrics:
            metric_info += f"        {metric}={metrics[metric]}\n"
        if "losses" in output:
            for loss in output["losses"]:
                metric_info += f"        {loss}={output['losses'][loss]}\n"

        print(f"{epoch_info} - {validation_info}")
        print(metric_info)


    print("Started Training")
    start = time.time()
    trainer.run(train_loader, max_epochs=num_epochs)
    end = time.time()
    duration = end - start
    print(f"Training took: {duration} s (this is an average of {duration / trainer.state.epoch} s per epoch)")

    """
    Test
    """

    evaluator.run(test_loader)
    metrics = evaluator.state.metrics
    output = evaluator.state.output
    print(f"Test Loss: {output['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

    if not my_setup.TRIPLET_MODE:
        for loader in [train_loader, test_loader]:
            loader_list = list(loader)

            test_seqs, labels, masks = loader_list[0]
            test_seqs = test_seqs.to(my_setup.DEVICE_TRAINING)
            labels = labels.to(my_setup.DEVICE_TRAINING)
            masks = masks.to(my_setup.DEVICE_TRAINING)

            model_output = encoder_decoder_model(test_seqs)
            if isinstance(model_output, MaskedLMOutput):
                model_output = model_output[0]
            axis = 2  # Needs to be 2 if output is of shape 14 x 5
            # axis = 1  # Needs to be 1 if output is of shape 5 x 14
            prediction = torch.argmax(torch.softmax(model_output, axis=axis), axis=axis)

            difference = labels - prediction

            loss = masked_cross_entropy_loss(model_output, loader_list[0][1])

            num_errors = torch.count_nonzero(difference)
            num_bases = difference.numel()
            masked_num_errors = torch.count_nonzero(difference[masks])
            masked_num_bases = torch.count_nonzero(masks)


            print("Diff:")
            print(difference)
            print("Num Sequences:", len(test_seqs))
            print("Num Errors:", num_errors)
            print("Fraction E:", num_errors / num_bases)
            print("Masked Num Errors:", masked_num_errors)
            print("Masked Fraction E:", masked_num_errors / masked_num_bases)
            print("Masked Cross Entropy Loss:", loss)
    pass
