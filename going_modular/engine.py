"""
Contains functions for training and testing a PyTorch model.
"""
import torch
import torch.nn.functional as f
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model1: torch.nn.Module, 
               model2: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               vit_optimizer: torch.optim.Optimizer,
               device: torch.device,
               batch_size:int) -> Tuple[float, float] :
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model1.train()
    model2.train()

    # Setup train loss and train accuracy values
    vit_train_loss, vit_train_acc = 0, 0
    weight_train_loss, weight_train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)
        # print(f"y shape:{y.shape}")
        # padding=torch.zeros((batch_size-y.shape[0],y.shape[1],y.shape[2]))        
        # print(f"padding shape:{padding.shape}")
        # y = f.pad(y,pad=(0,0,0,0,0,padding.shape[0]))
        
        # 1. Forward pass  2 models
        y_pred,vit_encoder_output = model1(X)

        weights = model2(vit_encoder_output)
        
        weight_optimizer = torch.optim.Adam(params=model2.parameters(),
                            lr=0.001,
                            betas=(0.9,0.999),
                            weight_decay=0.1)
        
        #  # Ensure weights have the same batch size as encoder_output
        # if weights.shape[0] != encoder_output.shape[0]:
        #     weights = weights[:encoder_output.shape[0]]
            
        # Apply weights to encoder output
        # print(f"Encoder output shape:{encoder_output.shape}")
        
        #batch matrix muktiplication using einsum
        weighted_output = torch.einsum('b p e, e p b -> b p e', vit_encoder_output, weights.transpose(0,2))
        # weighted_output = weights
        # print(f"Weight shape:{weighted_output.shape}")
        
       
        
        
        # 2. Calculate  and accumulate loss
        
        # print(f"y_pred_class shape:{y_pred_class.shape}")
        # print(f"y shape:{y.shape}")
        
        # padding=torch.zeros((batch_size-y.shape[0],y.shape[1],y.shape[2]))    
            
        y = f.pad(y,pad=(0,batch_size-y.shape[0]))
        vit_loss = loss_fn(y_pred, y)
        # print("weight output shape:",weighted_output.shape)
        # print("vit encoder output shape:",vit_encoder_output.shape)
        weight_loss = loss_fn(weighted_output, vit_encoder_output)
        total_loss = vit_loss + weight_loss
        vit_train_loss += vit_loss.item() 
        weight_train_loss += weight_loss.item() 

        # 3. Optimizer zero grad
        vit_optimizer.zero_grad()
        weight_optimizer.zero_grad()
        # 4. Loss backward
        # vit_loss.backward()
        # weight_loss.backward(retain_graph=True)
        total_loss.backward()
        # 5. Optimizer step
        vit_optimizer.step()
        weight_optimizer.step()
        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        vit_train_acc += (y_pred_class == y).sum().item()/len(y_pred)
        weight_train_acc += (weighted_output == vit_encoder_output).sum().item()/len(weighted_output)
        

    # Adjust metrics to get average loss and accuracy per batch 
    vit_train_loss = vit_train_loss / len(dataloader)
    weight_train_loss = weight_train_loss / len(dataloader)
    vit_train_acc = vit_train_acc / len(dataloader)
    weight_train_acc = weight_train_acc / len(dataloader)
    return vit_train_loss, weight_train_loss, vit_train_acc, weight_train_acc

def test_step(model1: torch.nn.Module,
              model2: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device,
              batch_size:int) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model1.eval() 
    model2.eval()

    # Setup test loss and test accuracy values
    vit_test_loss, vit_test_acc = 0, 0
    weight_test_loss, weight_test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
            y = f.pad(y,pad=(0,batch_size-y.shape[0]))
            # 1. Forward pass
            test_pred_logits,vit_encoder_output = model1(X)
            weights = model2(vit_encoder_output)
            
            
            # 2. Calculate and accumulate loss
            vit_loss = loss_fn(test_pred_logits, y)
            weight_loss = loss_fn(weights, vit_encoder_output)
            vit_test_loss += vit_loss.item()
            weight_test_loss += weight_loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            vit_test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            weight_test_acc += ((weights == vit_encoder_output).sum().item()/len(weights))

    # Adjust metrics to get average loss and accuracy per batch 
    vit_test_loss = vit_test_loss / len(dataloader)
    weight_test_loss = weight_test_loss / len(dataloader)
    vit_test_acc = vit_test_acc / len(dataloader)
    weight_test_acc = weight_test_acc / len(dataloader)
    return vit_test_loss, weight_test_loss, vit_test_acc, weight_test_acc

def train(model1: torch.nn.Module, 
          model2: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          vit_optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          batch_size:int,
          device: torch.device) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]} 
    For example if training for epochs=2: 
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]} 
    """
    # Create empty results dictionary
    results = {"vit_train_loss": [],
               "vit_train_acc": [],
               "weight_train_loss": [],
               "weight_train_acc": [],
               "vit_test_loss": [],
               "vit_test_acc": [],
               "weight_test_loss": [],
               "weight_test_acc": []
    }
    
    # Make sure model on target device
    model1.to(device)
    model2.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        vit_train_loss, weight_train_loss, vit_train_acc, weight_train_acc = train_step(model1=model1,
                                          model2=model2,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          vit_optimizer=vit_optimizer,
                                          device=device,
                                          batch_size=batch_size)
        vit_test_loss, weight_test_loss, vit_test_acc, weight_test_acc = test_step(model1=model1,
          model2=model2,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device,
          batch_size=batch_size)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"vit_train_loss: {vit_train_loss:.4f} | "
          f"vit_train_acc: {vit_train_acc:.4f} | "
          f"weight_train_loss: {weight_train_loss:.4f} | "
          f"weight_train_acc: {weight_train_acc:.4f} | "
          f"vit_test_loss: {vit_test_loss:.4f} | "
          f"vit_test_acc: {vit_test_acc:.4f} | "
          f"weight_test_loss: {weight_test_loss:.4f} | "
          f"weight_test_acc: {weight_test_acc:.4f}"
        )

        # Update results dictionary
        results["vit_train_loss"].append(vit_train_loss)
        results["vit_train_acc"].append(vit_train_acc)
        results["weight_train_loss"].append(weight_train_loss)
        results["weight_train_acc"].append(weight_train_acc)
        results["vit_test_loss"].append(vit_test_loss)
        results["vit_test_acc"].append(vit_test_acc)
        results["weight_test_loss"].append(weight_test_loss)
        results["weight_test_acc"].append(weight_test_acc)

    # Return the filled results at the end of the epochs
    return results
