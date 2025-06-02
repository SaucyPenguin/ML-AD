import pickle
from pybedtools import BedTool
from collections import defaultdict


def get_informative(top_indices, top_feature_scores, snp_list, snp_ref):
    """
    Selects informative SNPs based on top indices and scores.

    Parameters:
        top_indices (list): Indices of top features.
        top_feature_scores (list): Corresponding importance scores.
        snp_list (list): List of SNP identifiers.
        snp_ref (dict): Dictionary mapping SNPs to their reference alleles.

    Returns:
        dict: Dictionary of informative SNPs and their scores.
    """
    features_and_scores = {}
    important_snps = []
    for i in top_indices:
        snp_name = snp_list[i]
        ref = snp_ref.get(snp_name)
        rsid = ref["rsID"]
        features_and_scores[rsid] = top_feature_scores[i]
        important_snps.append(rsid)
    with open("features_and_scores.pkl", "wb") as f3:
        pickle.dump(features_and_scores, f3)
    return important_snps


def generate_bed(important_snps, snp_ref):
    """
    Generates a .bed-format list of SNPs using reference data.

    Parameters:
        important_snps (dict): Dictionary of informative SNPs.
        snp_ref (dict): Dictionary mapping SNPs to reference data.

    Returns:
        list: BED-format SNP entries for output or further processing.
    """
    with open("important_snps.bed", "w") as f:
        for illumina_id, ref in snp_ref.items():
            if ref is not None and ref["rsID"] in important_snps:
                chrom = ref["chromosome"]
                pos = ref["position"]
                rs_id = ref["rsID"]
                f.write(f"chr{chrom}\t{pos}\t{pos + 1}\t{rs_id}\n")


def cluster(filename):
    """
    Clusters SNPs based on BED-formatted input.

    Parameters:
        filename (str): Path to the BED-format file.

    Returns:
        dict: Mapping from cluster IDs to lists of SNPs.
    """
    bed = BedTool(filename)
    clusters = bed.sort().cluster(d=50000)
    clusters.saveas("snp_clusters.bed")
    cluster_map = defaultdict(list)
    with open("snp_clusters.bed") as f:
        for line in f:
            chrom, start, end, snp_id, cluster_id = line.strip().split()
            cluster_map[cluster_id].append({
                "chromosome": chrom,
                "start": start,
                "end": end,
                "snp_id": snp_id
            })
    with open("cluster_map.pkl", "wb") as f1:
        pickle.dump(cluster_map, f1)
    return cluster_map


def find_important_clusters(cluster_map, features_and_scores):
    """
    Identifies clusters containing highly important features.

    Parameters:
        cluster_map (dict): Cluster ID to SNP list mapping.
        features_and_scores (dict): SNP importance scores.

    Returns:
        dict: Cluster scores indicating relative importance.
    """
    cluster_scores_dict = defaultdict(list)
    for cluster_id, snps in cluster_map.items():
        for snp in snps:
            snp_id = snp["snp_id"]
            if snp_id in features_and_scores:
                cluster_scores_dict[cluster_id].append(features_and_scores[snp_id])
            else:
                print(f"⚠️ Missing SNP in features_and_scores: {snp_id}")
    with open("cluster_scores_dict.pkl", "wb") as f:
        pickle.dump(cluster_scores_dict, f)
    return cluster_scores_dict


def heuristic(snp_cluster, alpha, beta):
    """
    Computes a weighted max-mean heuristic score for a SNP cluster.

    Parameters:
        snp_cluster (list): List of SNPs in the cluster.
        alpha (float): Weight for SNP count.
        beta (float): Weight for cumulative feature score.

    Returns:
        float: Heuristic score representing cluster importance.
    """
    if len(snp_cluster) == 1:
        return snp_cluster[0]
    sorted_scores = sorted(snp_cluster, reverse=True)
    return (alpha * sorted_scores[0]) + ((sum(sorted_scores[1::]) / (len(sorted_scores) - 1)) * beta)


def rank_clusters(clusters_scores_dict):
    """
    Ranks SNP clusters based on their heuristic scores.

    Parameters:
        clusters_scores_dict (dict): Mapping of cluster IDs to scores.

    Returns:
        list: Sorted list of clusters by descending importance.
    """
    cluster_scores = []
    clusters_and_score = {}
    for cluster_id, scores in clusters_scores_dict.items():
        cluster_score = heuristic(scores, 0.8, 0.2)
        cluster_scores.append(cluster_score)
        clusters_and_score[cluster_id] = cluster_score
    ranked_clusters = [i[1] for i in sorted(zip(cluster_scores, list(clusters_scores_dict.keys())), reverse=True)]
    with open("ranked_clusters.pkl", "wb") as f:
        pickle.dump(ranked_clusters, f)
    with open("clusters_and_score.pkl", "wb") as f1:
        pickle.dump(clusters_and_score, f1)
    return ranked_clusters, clusters_and_score
