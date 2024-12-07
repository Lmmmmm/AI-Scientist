import numpy as np
import pandas as pd
import torch

def generate_base_data(n_samples):
    """Generate base urban planning data"""
    return {
        # Land Use Data (4 dimensions)
        'land_use_type': np.random.choice([1, 2, 3], n_samples),  # 1=residential, 2=mixed, 3=commercial
        'building_density': np.clip(np.random.normal(0.62, 0.05, n_samples), 0, 1),
        'floor_area_ratio': np.clip(np.random.normal(2.0, 0.5, n_samples), 0, 5),
        'vacant_land_ratio': np.clip(np.random.normal(0.15, 0.05, n_samples), 0, 1),

        # Facility Data (4 dimensions)
        'medical_coverage': np.clip(np.random.normal(0.8, 0.1, n_samples), 0, 1),
        'education_coverage': np.clip(np.random.normal(0.9, 0.1, n_samples), 0, 1),
        'transport_coverage': np.clip(np.random.normal(0.85, 0.1, n_samples), 0, 1),
        'community_coverage': np.clip(np.random.normal(0.7, 0.1, n_samples), 0, 1),

        # Socioeconomic Data (4 dimensions)
        'population_density': np.random.normal(15000, 2000, n_samples),
        'income_level': np.random.normal(450, 50, n_samples),
        'marriage_rate': np.clip(np.random.normal(0.45, 0.05, n_samples), 0, 1),
        'birth_rate': np.clip(np.random.normal(0.12, 0.02, n_samples), 0, 1),

        # Cost Data (4 dimensions)
        'housing_cost': np.random.normal(95, 15, n_samples),
        'rental_cost': np.random.normal(30, 5, n_samples),
        'subsidy_coverage': np.clip(np.random.normal(0.7, 0.1, n_samples), 0, 1),
        'maintenance_cost': np.random.normal(50, 10, n_samples)
    }

def get_urban_dataset(n_samples=1000):
    """Generate dataset for model training"""
    data = generate_base_data(n_samples)
    tensor_data = torch.zeros((n_samples, 16))

    # Convert dictionary data to tensor
    for i, key in enumerate(data.keys()):
        tensor_data[:, i] = torch.tensor(data[key])

    return torch.utils.data.TensorDataset(tensor_data)

# if __name__ == "__main__":
#     n_samples = 1000
#     dataset = get_urban_dataset(n_samples)

#     data_dict = generate_base_data(n_samples)
#     df = pd.DataFrame(data_dict)
#     df.to_csv('urban_planning_data.tsv', sep='\t', index=False)

#     print(f"Generated {n_samples} samples and saved to urban_planning_data.tsv")
#     print("Dataset shape:", dataset.tensors[0].shape)