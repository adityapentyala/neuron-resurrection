import torch
import torch.nn as nn
import torch.optim as optim
from model import ResurrectionNetwork, get_neuron_importance, resurrect_layer
from dataloader import get_task_dataloaders
import matplotlib.pyplot as plt
import copy

def train_task(model, task_id, train_loader, epochs=3, save_path=None):
    model.train()
    # We use two optimizers: one for the backbone, one for the probes
    optimizer_main = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"--- Training Task {task_id} (Digits {task_id*2}, {task_id*2+1}) ---")
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer_main.zero_grad()
            
            # Forward pass with probes enabled
            output, probe1_output, probe2_output, probe3_output= model(data, probe=True)
            
            # 1. Main Loss (only on the relevant task digits)
            loss_main = criterion(output, target)
            
            # 2. Probe Loss (training probes to recognize current task digits)
            loss_probes = sum(criterion(p_out, target) for p_out in [probe1_output, probe2_output, probe3_output])
            
            # Total loss (Probes only update their own weights because of .detach() in model)
            combined_loss = loss_main + loss_probes
            combined_loss.backward()
            
            optimizer_main.step()
            total_loss += combined_loss.item()
            
        print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}")
    if save_path:
        torch.save(model.state_dict(), f"checkpoints/{save_path}_task{task_id}.pth")

def evaluate_task(model, task_id, test_loader):
    model.eval()
    correct = 0
    probe_correct = [0, 0, 0] # Accuracy for [Probe 1, Probe 2, Probe 3]
    class_accuracy = {i: [0, 0] for i in range(10)}  # {class: [correct, total]}
    
    with torch.no_grad():
        for data, target in test_loader:
            output, probe1_output, probe2_output, probe3_output= model(data, probe=True)
            
            # Main model accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Probes accuracy
            probe_outputs = [probe1_output, probe2_output, probe3_output]
            for i, p_out in enumerate(probe_outputs):
                p_pred = p_out.argmax(dim=1, keepdim=True)
                probe_correct[i] += p_pred.eq(target.view_as(p_pred)).sum().item()

            # Class-wise accuracy
            for t, p in zip(target.view(-1), pred.view(-1)):
                class_accuracy[t.item()][1] += 1
                if t.item() == p.item():
                    class_accuracy[t.item()][0] += 1
                #print(class_accuracy)
            
    plt.figure(figsize=(10,5))
    plt.bar(class_accuracy.keys(), [c[0]/c[1] if c[1] > 0 else 0 for c in class_accuracy.values()])
    plt.xlabel('Digit Class')   
    plt.ylabel('Accuracy')
    plt.title(f'Class-wise Accuracy for Task {task_id}')
    plt.xticks(range(10))
    plt.ylim(0, 1)
    plt.savefig(f"results/task{task_id}_class_accuracy.png")       
    n = len(test_loader.dataset)
    print(f"Task {task_id} Results - Main: {100.*correct/n:.2f}% | Probe1: {100.*probe_correct[0]/n:.2f}% | Probe2: {100.*probe_correct[1]/n:.2f}% | Probe3: {100.*probe_correct[2]/n:.2f}%")


if __name__ == "__main__":

    resurrect = True
    zero_grad = True
    top_percent = 0.15
    
    model = ResurrectionNetwork(input_size=28*28, output_size=10)
    
    for task_id in range(5):  # 5 tasks: (0,1), (2,3), (4,5), (6,7), (8,9)
        train_loader, _ = get_task_dataloaders(task_id)
        test_loader = get_task_dataloaders(task_id, test_only=True, filter=list(range(0, task_id*2+2)))
        train_task(model, task_id, train_loader, epochs=3, save_path="resurrection_model")

        # resurrect
        if resurrect and task_id > 0:
            prev_model= ResurrectionNetwork(input_size=28*28, output_size=10)
            prev_model.load_state_dict(torch.load(f"checkpoints/resurrection_model_task{task_id-1}.pth"))
            # Calculate importance for each hidden layer
            importance_l1 = get_neuron_importance(prev_model.fc1)
            importance_l2 = get_neuron_importance(prev_model.fc2)
            importance_l3 = get_neuron_importance(prev_model.fc3)
            importance_l4 = get_neuron_importance(prev_model.fc4)
            # Resurrect top 20% of neurons in each layer based on Task 1 importance
            resurrect_layer(model, prev_model, importance_l1, 'fc1', top_percent=top_percent*task_id, zero_grad=zero_grad)
            resurrect_layer(model, prev_model, importance_l2, 'fc2', top_percent=top_percent*task_id, zero_grad=zero_grad)
            resurrect_layer(model, prev_model, importance_l3, 'fc3', top_percent=top_percent*task_id, zero_grad=zero_grad)
            resurrect_layer(model, prev_model, importance_l4, 'fc4', top_percent=1.0, zero_grad=zero_grad)

        evaluate_task(model, task_id, test_loader)

        # Post-Task Checkpoint ---
        print(f"Saving Task {task_id} snapshot and calculating importance...")

        # Deepcopy is crucial here so the saved state doesn't update with the model
        #task_snapshot = copy.deepcopy(model.state_dict())

        

        