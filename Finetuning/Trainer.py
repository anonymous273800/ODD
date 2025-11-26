from LossHandler import LossCalculator

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = LossCalculator.calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()  # Returns the total number of elements (or tokens) in the input_batch.
            global_step += 1

            # # Optional evaluation step
            # if global_step % eval_freq == 0:
            #     train_loss, val_loss = LossCalculator.evaluate_model(
            #         model, train_loader, val_loader, device, eval_iter)
            #     train_losses.append(train_loss)
            #     val_losses.append(val_loss)
            #     track_tokens_seen.append(tokens_seen)
            #     print(f"Ep {epoch + 1} (Step {global_step:06d}): "
            #           f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

    return model, train_losses, val_losses, track_tokens_seen