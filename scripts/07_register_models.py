import os
import mlflow
from mlflow.tracking import MlflowClient

# Set the remote MLflow Tracking URI
mlflow_tracking_uri = "http://13.58.96.234:5000"  # Replace with your EC2 public IP
mlflow.set_tracking_uri(mlflow_tracking_uri)
print(f"MLFLOW_TRACKING_URI: {mlflow_tracking_uri}")

MODEL_SPECS = [
    {"name": "steel_energy_gbm", "tag_run_name": "h2o_gbm"},
    {"name": "steel_energy_dnn", "tag_run_name": "h2o_dnn"},
    {"name": "steel_energy_randomforest", "tag_run_name": "random_forest_regressor"},
]

ARTIFACT_PATH = "model"  # change if you logged a different artifact path

def get_latest_run_id(client: MlflowClient, experiment_name: str, run_name_tag: str):
    exp = client.get_experiment_by_name(experiment_name)
    if not exp:
        raise RuntimeError(f"Experiment '{experiment_name}' not found.")

    # Search by tag instead of run.info.run_name
    filter_str = f"tags.mlflow.runName = '{run_name_tag}'"
    runs = client.search_runs(exp.experiment_id, filter_string=filter_str, order_by=["start_time DESC"])
    if not runs:
        raise RuntimeError(f"No run found with tag mlflow.runName = '{run_name_tag}' in experiment '{experiment_name}'")

    return runs[0].info.run_id

def ensure_registered(client: MlflowClient, model_name: str, run_id: str, artifact_path: str):
    # If registered model exists, we’ll still register a new version from the run
    model_uri = f"runs:/{run_id}/{artifact_path}"
    result = mlflow.register_model(model_uri, model_name)
    print(f"Registered {model_name} from run {run_id} as version {result.version}")
    return result.version

def main():
    client = MlflowClient()
    experiment_name = "steel_energy_models"

    assigned_versions = {}

    # 1) Register models
    for spec in MODEL_SPECS:
        model_name = spec["name"]
        run_id = get_latest_run_id(client, experiment_name, spec["tag_run_name"])
        if not run_id:
            raise RuntimeError(f"No suitable run found for {model_name} (expected run tag '{spec['tag_run_name']}').")
        version = ensure_registered(client, model_name, run_id, ARTIFACT_PATH)
        assigned_versions[model_name] = version

    # 2) Update descriptions for all versions
    for spec in MODEL_SPECS:
        name = spec["name"]
        versions = client.search_model_versions(f"name='{name}'")
        for v in versions:
            client.update_model_version(
                name, v.version, description=f"Auto-registered version {v.version} for {name}"
            )

    # 3) Assign aliases (preferred over stages)
    # Champion: GBM
    gbm_version = assigned_versions.get("steel_energy_gbm")
    if gbm_version:
        client.set_registered_model_alias("steel_energy_gbm", "champion", gbm_version)
        print(f"Alias set: steel_energy_gbm@champion -> v{gbm_version}")

    # Staging: DNN, RF
    for name in ["steel_energy_dnn", "steel_energy_randomforest"]:
        v = assigned_versions.get(name)
        if v:
            client.set_registered_model_alias(name, "staging", v)
            print(f"Alias set: {name}@staging -> v{v}")

    print("Model registration complete. Aliases assigned.")

if __name__ == "__main__":
    main()