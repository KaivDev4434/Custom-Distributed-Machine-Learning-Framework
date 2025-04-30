import numpy as np
import time
import os
from datetime import datetime
from mpi_wrapper import init_mpi, sync_gradients, finalize_mpi, get_world_size, get_world_rank
from nn_model import Model  # Your CUDA model
from data_loader import load_data  # Your data loader

class DistributedTrainer:
    def __init__(self, batch_size=32, num_epochs=10, learning_rate=0.01):
        # Initialize MPI
        init_mpi()
        self.rank = get_world_rank()
        self.world_size = get_world_size()
        
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        # Initialize model and metrics
        self.model = Model()
        self.train_losses = []
        self.train_accuracies = []
        
    def load_and_partition_data(self):
        # Load and partition data across processes
        train_data, train_labels = load_data()
        num_samples = len(train_data)
        samples_per_process = num_samples // self.world_size
        
        start_idx = self.rank * samples_per_process
        end_idx = start_idx + samples_per_process if self.rank != self.world_size - 1 else num_samples
        
        self.local_data = train_data[start_idx:end_idx]
        self.local_labels = train_labels[start_idx:end_idx]
        
        if self.rank == 0:
            print(f"Total samples: {num_samples}")
            print(f"Samples per process: {samples_per_process}")
    
    def train(self):
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            # Shuffle data
            indices = np.random.permutation(len(self.local_data))
            self.local_data = self.local_data[indices]
            self.local_labels = self.local_labels[indices]
            
            num_batches = (len(self.local_data) + self.batch_size - 1) // self.batch_size
            
            for i in range(0, len(self.local_data), self.batch_size):
                batch_data = self.local_data[i:i+self.batch_size]
                batch_labels = self.local_labels[i:i+self.batch_size]
                
                # Forward pass
                predictions = self.model.forward(batch_data)
                
                # Compute loss and gradients
                loss, gradients = self.model.backward(predictions, batch_labels)
                
                # Synchronize gradients across all processes
                try:
                    sync_gradients(gradients)
                except Exception as e:
                    print(f"Error in gradient synchronization: {e}")
                    finalize_mpi()
                    raise
                
                # Update model parameters
                self.model.update_parameters(gradients, self.learning_rate)
                
                # Update metrics
                epoch_loss += loss
                predicted = np.argmax(predictions, axis=1)
                correct += np.sum(predicted == batch_labels)
                total += len(batch_labels)
                
                # Print progress (from rank 0 only)
                if self.rank == 0 and (i // self.batch_size) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.num_epochs}, "
                          f"Batch {i//self.batch_size + 1}/{num_batches}, "
                          f"Loss: {loss:.4f}")
            
            # Compute and log epoch metrics
            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / num_batches
            accuracy = correct / total
            
            self.train_losses.append(avg_loss)
            self.train_accuracies.append(accuracy)
            
            if self.rank == 0:
                print(f"\nEpoch {epoch+1} Summary:")
                print(f"Average Loss: {avg_loss:.4f}")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Time: {epoch_time:.2f}s")
                print("-" * 50)
        
        # Save model and training history (from rank 0 only)
        if self.rank == 0:
            total_time = time.time() - start_time
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(model_dir, f"model_{timestamp}.npy")
            history_path = os.path.join(model_dir, f"history_{timestamp}.npz")
            
            self.model.save(model_path)
            np.savez(history_path, 
                     losses=self.train_losses,
                     accuracies=self.train_accuracies)
            
            print(f"\nTraining completed in {total_time:.2f} seconds")
            print(f"Model saved to: {model_path}")
            print(f"Training history saved to: {history_path}")
        
        finalize_mpi()

def main():
    # Training hyperparameters
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.01
    
    # Initialize and run distributed training
    trainer = DistributedTrainer(batch_size, num_epochs, learning_rate)
    trainer.load_and_partition_data()
    trainer.train()

if __name__ == "__main__":
    main() 