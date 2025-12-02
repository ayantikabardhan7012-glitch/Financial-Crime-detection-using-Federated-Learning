import flwr as fl
import pickle
import os
from flwr.server.strategy import FedAvg

SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

#Saves model each round using FedAvg
class SaveModelFedAvg(FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        """Aggregate fit results and save the global model after each round."""
        aggregated_weights, aggregated_metrics = super().aggregate_fit(rnd, results, failures)

        save_path = os.path.join(SAVE_DIR, f"global_model_round_{rnd}.pkl")

        # Save global model weights as pickle
        with open(save_path, "wb") as f:
            pickle.dump(aggregated_weights, f)
        print(f"[Server] Saved global model from round {rnd}.")
        
        return aggregated_weights, aggregated_metrics

#Strategy instance
strategy = SaveModelFedAvg(
    fraction_fit=1.0,                 # use all available clients for training
    fraction_evaluate=1.0,            # use all clients for evaluation
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
    evaluate_metrics_aggregation_fn=lambda results: {
        # Weighted average of accuracy
        "accuracy": sum(metrics["accuracy"] * num_examples for num_examples, metrics in results)
                    / sum(num_examples for num_examples, _ in results)
        if len(results) > 0 else 0.0
    },
)

if __name__ == "__main__":
    fl.server.start_server(
        server_address="localhost:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=15),  # number of FL rounds
    )
