import argparse
import torch
from models.puzzle_solver import PuzzleSolver
from data.datasets.multi_domain_dataset import MultiDomainDataset
from utils.metrics import calculate_accuracy, calculate_psnr, calculate_ssim

def evaluate(model, dataloader, device):
    model.eval()
    total_accuracy = 0
    total_top_k_accuracy = 0
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    total_semantic_consistency = 0
    num_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            image_patches, labels = batch
            image_patches = image_patches.to(device)
            labels = labels.to(device)

            position_logits, relation_logits, reconstructed_image = model(image_patches)

            accuracy = calculate_accuracy(position_logits, labels)
            top_k_accuracy = accuracy_metrics.compute_top_k_accuracy(position_logits, labels)
            psnr = calculate_psnr(reconstructed_image, image_patches)
            ssim = calculate_ssim(reconstructed_image, image_patches)
            lpips = image_quality_metrics.compute_lpips(reconstructed_image, image_patches)
            semantic_consistency = semantic_metrics.compute_semantic_consistency(reconstructed_image, image_patches)

            total_accuracy += accuracy
            total_top_k_accuracy += top_k_accuracy
            total_psnr += psnr
            total_ssim += ssim
            total_lpips += lpips
            total_semantic_consistency += semantic_consistency
            num_samples += 1

    avg_accuracy = total_accuracy / num_samples
    avg_top_k_accuracy = total_top_k_accuracy / num_samples
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    avg_lpips = total_lpips / num_samples
    avg_semantic_consistency = total_semantic_consistency / num_samples

    return {
        'accuracy': avg_accuracy,
        'top_k_accuracy': avg_top_k_accuracy,
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'lpips': avg_lpips,
        'semantic_consistency': avg_semantic_consistency
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate the puzzle solver model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the evaluation dataset.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the evaluation on.')
    args = parser.parse_args()

    device = torch.device(args.device)
    model = PuzzleSolver()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    dataset = MultiDomainDataset(args.data_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    metrics = evaluate(model, dataloader, device)

    for metric, value in metrics.items():
        print(f'{metric}: {value}')

if __name__ == '__main__':
    main()
