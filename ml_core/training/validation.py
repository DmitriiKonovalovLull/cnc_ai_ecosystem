import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ModelValidator:
    def __init__(self, validation_dir: Optional[Path] = None):
        self.validation_dir = validation_dir or Path(__file__).parent.parent.parent / "data" / "validation"
        self.validation_dir.mkdir(parents=True, exist_ok=True)

        # Validation metrics storage
        self.metrics_history = {}

    def validate_intent_classifier(self, predictions: List[str],
                                   ground_truth: List[str]) -> Dict[str, Any]:
        """Validate intent classifier predictions"""
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support,
            confusion_matrix, classification_report
        )

        try:
            # Calculate metrics
            accuracy = accuracy_score(ground_truth, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                ground_truth, predictions, average='weighted'
            )

            # Confusion matrix
            cm = confusion_matrix(ground_truth, predictions)
            unique_labels = sorted(set(ground_truth + predictions))

            # Classification report
            report = classification_report(
                ground_truth, predictions,
                target_names=unique_labels,
                output_dict=True
            )

            # Calculate per-class metrics
            per_class_metrics = {}
            for label in unique_labels:
                if label in report:
                    per_class_metrics[label] = {
                        'precision': report[label]['precision'],
                        'recall': report[label]['recall'],
                        'f1': report[label]['f1-score'],
                        'support': report[label]['support']
                    }

            results = {
                'overall_metrics': {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1)
                },
                'per_class_metrics': per_class_metrics,
                'confusion_matrix': cm.tolist(),
                'class_labels': unique_labels,
                'samples_processed': len(predictions),
                'timestamp': datetime.now().isoformat()
            }

            # Save validation results
            self._save_validation_results('intent_classifier', results)

            logger.info(f"Intent classifier validation: Accuracy={accuracy:.4f}, F1={f1:.4f}")
            return results

        except Exception as e:
            logger.error(f"Intent classifier validation failed: {e}")
            raise

    def validate_ner_model(self, predictions: List[List[Dict[str, Any]]],
                           ground_truth: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Validate NER model predictions"""
        try:
            # Flatten predictions and ground truth
            pred_entities = self._flatten_entities(predictions)
            true_entities = self._flatten_entities(ground_truth)

            # Calculate entity-level metrics
            entity_metrics = self._calculate_entity_metrics(pred_entities, true_entities)

            # Calculate token-level metrics
            token_metrics = self._calculate_token_metrics(predictions, ground_truth)

            results = {
                'entity_level_metrics': entity_metrics,
                'token_level_metrics': token_metrics,
                'samples_processed': len(predictions),
                'timestamp': datetime.now().isoformat()
            }

            # Save validation results
            self._save_validation_results('ner_model', results)

            logger.info(f"NER model validation completed")
            return results

        except Exception as e:
            logger.error(f"NER model validation failed: {e}")
            raise

    def _flatten_entities(self, entity_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Flatten nested entity lists"""
        flat_entities = []
        for entities in entity_lists:
            flat_entities.extend(entities)
        return flat_entities

    def _calculate_entity_metrics(self, pred_entities: List[Dict[str, Any]],
                                  true_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate entity-level metrics"""
        # This is a simplified version
        # In production, you'd need exact boundary matching

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        # Simple exact match comparison
        pred_set = set()
        for ent in pred_entities:
            key = f"{ent.get('text', '')}_{ent.get('type', '')}_{ent.get('start', 0)}"
            pred_set.add(key)

        true_set = set()
        for ent in true_entities:
            key = f"{ent.get('text', '')}_{ent.get('type', '')}_{ent.get('start', 0)}"
            true_set.add(key)

        true_positives = len(pred_set.intersection(true_set))
        false_positives = len(pred_set - true_set)
        false_negatives = len(true_set - pred_set)

        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'total_entities': len(true_entities)
        }

    def _calculate_token_metrics(self, predictions: List[List[Dict[str, Any]]],
                                 ground_truth: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Calculate token-level metrics"""
        # Simplified token-level calculation
        total_tokens = 0
        correct_tokens = 0

        for pred_list, true_list in zip(predictions, ground_truth):
            # This would require actual tokenization and alignment
            # For now, return placeholder
            pass

        return {
            'accuracy': 0.0,
            'total_tokens': total_tokens,
            'correct_tokens': correct_tokens
        }

    def validate_answer_quality(self, generated_answers: List[str],
                                reference_answers: List[str]) -> Dict[str, Any]:
        """Validate answer quality using multiple metrics"""
        try:
            # BLEU score for text similarity
            bleu_scores = self._calculate_bleu_scores(generated_answers, reference_answers)

            # ROUGE scores
            rouge_scores = self._calculate_rouge_scores(generated_answers, reference_answers)

            # Semantic similarity using embeddings
            semantic_similarities = self._calculate_semantic_similarity(
                generated_answers, reference_answers
            )

            # Length analysis
            length_stats = self._analyze_lengths(generated_answers, reference_answers)

            results = {
                'bleu_scores': bleu_scores,
                'rouge_scores': rouge_scores,
                'semantic_similarity': semantic_similarities,
                'length_analysis': length_stats,
                'samples_processed': len(generated_answers),
                'timestamp': datetime.now().isoformat()
            }

            # Save validation results
            self._save_validation_results('answer_quality', results)

            logger.info(f"Answer quality validation completed")
            return results

        except Exception as e:
            logger.error(f"Answer quality validation failed: {e}")
            raise

    def _calculate_bleu_scores(self, generated: List[str], reference: List[str]) -> Dict[str, float]:
        """Calculate BLEU scores"""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

            scores = []
            smoothie = SmoothingFunction().method4

            for gen, ref in zip(generated, reference):
                # Tokenize
                gen_tokens = gen.split()
                ref_tokens = [ref.split()]

                # Calculate BLEU
                score = sentence_bleu(ref_tokens, gen_tokens, smoothing_function=smoothie)
                scores.append(score)

            return {
                'mean': float(np.mean(scores)) if scores else 0.0,
                'std': float(np.std(scores)) if scores else 0.0,
                'min': float(np.min(scores)) if scores else 0.0,
                'max': float(np.max(scores)) if scores else 0.0,
                'scores': [float(s) for s in scores]
            }

        except ImportError:
            logger.warning("NLTK not available for BLEU calculation")
            return {'mean': 0.0, 'note': 'NLTK not installed'}

    def _calculate_rouge_scores(self, generated: List[str], reference: List[str]) -> Dict[str, Any]:
        """Calculate ROUGE scores"""
        # This would require rouge-score library
        # For now, return placeholder
        return {
            'rouge1': {'f1': 0.0},
            'rouge2': {'f1': 0.0},
            'rougeL': {'f1': 0.0},
            'note': 'ROUGE calculation not implemented'
        }

    def _calculate_semantic_similarity(self, generated: List[str],
                                       reference: List[str]) -> Dict[str, float]:
        """Calculate semantic similarity using embeddings"""
        try:
            from sentence_transformers import SentenceTransformer

            # Load embedding model
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

            # Encode sentences
            gen_embeddings = model.encode(generated, convert_to_tensor=True)
            ref_embeddings = model.encode(reference, convert_to_tensor=True)

            # Calculate cosine similarities
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(gen_embeddings.cpu(), ref_embeddings.cpu())

            # Get diagonal (matching pairs)
            diag_similarities = similarities.diagonal()

            return {
                'mean': float(np.mean(diag_similarities)) if len(diag_similarities) > 0 else 0.0,
                'std': float(np.std(diag_similarities)) if len(diag_similarities) > 0 else 0.0,
                'min': float(np.min(diag_similarities)) if len(diag_similarities) > 0 else 0.0,
                'max': float(np.max(diag_similarities)) if len(diag_similarities) > 0 else 0.0,
                'similarities': [float(s) for s in diag_similarities]
            }

        except Exception as e:
            logger.error(f"Semantic similarity calculation failed: {e}")
            return {'mean': 0.0, 'error': str(e)}

    def _analyze_lengths(self, generated: List[str], reference: List[str]) -> Dict[str, Any]:
        """Analyze answer lengths"""
        gen_lengths = [len(ans.split()) for ans in generated]
        ref_lengths = [len(ans.split()) for ans in reference]

        # Calculate length ratios
        ratios = []
        for gen_len, ref_len in zip(gen_lengths, ref_lengths):
            if ref_len > 0:
                ratios.append(gen_len / ref_len)

        return {
            'generated': {
                'mean': float(np.mean(gen_lengths)) if gen_lengths else 0.0,
                'std': float(np.std(gen_lengths)) if gen_lengths else 0.0,
                'min': int(np.min(gen_lengths)) if gen_lengths else 0,
                'max': int(np.max(gen_lengths)) if gen_lengths else 0
            },
            'reference': {
                'mean': float(np.mean(ref_lengths)) if ref_lengths else 0.0,
                'std': float(np.std(ref_lengths)) if ref_lengths else 0.0,
                'min': int(np.min(ref_lengths)) if ref_lengths else 0,
                'max': int(np.max(ref_lengths)) if ref_lengths else 0
            },
            'length_ratios': {
                'mean': float(np.mean(ratios)) if ratios else 0.0,
                'std': float(np.std(ratios)) if ratios else 0.0,
                'min': float(np.min(ratios)) if ratios else 0.0,
                'max': float(np.max(ratios)) if ratios else 0.0
            }
        }

    def validate_retrieval_system(self, retrieved_docs: List[List[Dict[str, Any]]],
                                  relevant_docs: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Validate retrieval system performance"""
        try:
            # Calculate retrieval metrics
            precisions_at_k = []
            recalls_at_k = []
            ndcgs_at_k = []

            k_values = [1, 3, 5, 10]

            for k in k_values:
                prec_k, rec_k, ndcg_k = self._calculate_retrieval_metrics(
                    retrieved_docs, relevant_docs, k
                )
                precisions_at_k.append(prec_k)
                recalls_at_k.append(rec_k)
                ndcgs_at_k.append(ndcg_k)

            # Calculate Mean Average Precision (MAP)
            map_score = self._calculate_map(retrieved_docs, relevant_docs)

            # Calculate Mean Reciprocal Rank (MRR)
            mrr_score = self._calculate_mrr(retrieved_docs, relevant_docs)

            results = {
                'precision_at_k': {
                    f'P@{k}': float(score)
                    for k, score in zip(k_values, precisions_at_k)
                },
                'recall_at_k': {
                    f'R@{k}': float(score)
                    for k, score in zip(k_values, recalls_at_k)
                },
                'ndcg_at_k': {
                    f'NDCG@{k}': float(score)
                    for k, score in zip(k_values, ndcgs_at_k)
                },
                'map': float(map_score),
                'mrr': float(mrr_score),
                'queries_processed': len(retrieved_docs),
                'timestamp': datetime.now().isoformat()
            }

            # Save validation results
            self._save_validation_results('retrieval_system', results)

            logger.info(f"Retrieval system validation: MAP={map_score:.4f}, MRR={mrr_score:.4f}")
            return results

        except Exception as e:
            logger.error(f"Retrieval system validation failed: {e}")
            raise

    def _calculate_retrieval_metrics(self, retrieved: List[List[Dict[str, Any]]],
                                     relevant: List[List[Dict[str, Any]]], k: int) -> tuple:
        """Calculate retrieval metrics at cutoff k"""
        precisions = []
        recalls = []
        ndcgs = []

        for ret_list, rel_list in zip(retrieved, relevant):
            # Get top-k retrieved documents
            top_k = ret_list[:k]

            # Get relevant document IDs
            rel_ids = {doc.get('id', '') for doc in rel_list}

            # Calculate precision and recall
            relevant_retrieved = sum(1 for doc in top_k if doc.get('id', '') in rel_ids)
            precision = relevant_retrieved / k if k > 0 else 0
            recall = relevant_retrieved / len(rel_ids) if len(rel_ids) > 0 else 0

            precisions.append(precision)
            recalls.append(recall)

            # Calculate NDCG
            ndcg = self._calculate_ndcg(top_k, rel_list, k)
            ndcgs.append(ndcg)

        # Average metrics
        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0
        avg_ndcg = np.mean(ndcgs) if ndcgs else 0

        return avg_precision, avg_recall, avg_ndcg

    def _calculate_ndcg(self, retrieved: List[Dict[str, Any]],
                        relevant: List[Dict[str, Any]], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain"""
        # Simplified NDCG calculation
        rel_dict = {doc.get('id', ''): doc.get('relevance', 1.0) for doc in relevant}

        # Calculate DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved[:k]):
            rel = rel_dict.get(doc.get('id', ''), 0.0)
            dcg += rel / np.log2(i + 2)  # i+2 because i starts from 0

        # Calculate ideal DCG
        ideal_relevances = sorted(rel_dict.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))

        return dcg / idcg if idcg > 0 else 0.0

    def _calculate_map(self, retrieved: List[List[Dict[str, Any]]],
                       relevant: List[List[Dict[str, Any]]]) -> float:
        """Calculate Mean Average Precision"""
        aps = []

        for ret_list, rel_list in zip(retrieved, relevant):
            rel_ids = {doc.get('id', '') for doc in rel_list}

            # Calculate precision at each relevant document
            precisions = []
            relevant_count = 0

            for i, doc in enumerate(ret_list):
                if doc.get('id', '') in rel_ids:
                    relevant_count += 1
                    precision_at_i = relevant_count / (i + 1)
                    precisions.append(precision_at_i)

            # Calculate average precision for this query
            ap = np.mean(precisions) if precisions else 0
            aps.append(ap)

        return np.mean(aps) if aps else 0.0

    def _calculate_mrr(self, retrieved: List[List[Dict[str, Any]]],
                       relevant: List[List[Dict[str, Any]]]) -> float:
        """Calculate Mean Reciprocal Rank"""
        reciprocal_ranks = []

        for ret_list, rel_list in zip(retrieved, relevant):
            rel_ids = {doc.get('id', '') for doc in rel_list}

            # Find rank of first relevant document
            first_relevant_rank = None
            for i, doc in enumerate(ret_list):
                if doc.get('id', '') in rel_ids:
                    first_relevant_rank = i + 1
                    break

            if first_relevant_rank:
                reciprocal_ranks.append(1.0 / first_relevant_rank)
            else:
                reciprocal_ranks.append(0.0)

        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    def _save_validation_results(self, model_type: str, results: Dict[str, Any]):
        """Save validation results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"validation_{model_type}_{timestamp}.json"
        filepath = self.validation_dir / filename

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            # Update metrics history
            if model_type not in self.metrics_history:
                self.metrics_history[model_type] = []

            self.metrics_history[model_type].append({
                'timestamp': timestamp,
                'metrics': results.get('overall_metrics') or results.get('entity_level_metrics') or {}
            })

            logger.debug(f"Validation results saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save validation results: {e}")

    def get_validation_history(self, model_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get validation history for a model type"""
        history = self.metrics_history.get(model_type, [])
        return history[-limit:] if limit else history

    def generate_validation_report(self, model_type: str) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        history = self.get_validation_history(model_type, limit=20)

        if not history:
            return {'status': 'no_history', 'model_type': model_type}

        # Aggregate metrics
        aggregated = {
            'model_type': model_type,
            'total_validations': len(history),
            'time_period': {
                'first': history[0]['timestamp'],
                'last': history[-1]['timestamp']
            },
            'metric_trends': {}
        }

        # Extract metric trends
        metrics_to_track = ['accuracy', 'precision', 'recall', 'f1', 'map', 'mrr']

        for metric in metrics_to_track:
            values = []
            timestamps = []

            for entry in history:
                if metric in entry['metrics']:
                    values.append(entry['metrics'][metric])
                    timestamps.append(entry['timestamp'])

            if values:
                aggregated['metric_trends'][metric] = {
                    'values': values,
                    'timestamps': timestamps,
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'trend': self._calculate_trend(values)
                }

        return aggregated

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return 'insufficient_data'

        # Simple linear trend
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)

        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable'