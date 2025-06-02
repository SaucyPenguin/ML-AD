import pandas as pd
from pybedtools import BedTool
import pickle
from collections import defaultdict


def gtf_to_bed(gtf_file):
    """
    Converts a GTF (Gene Transfer Format) file to BED format. Used for importing Ensembl gene annotations.

    Parameters:
        gtf_file (str): Path to the input GTF file.

    Returns:
        list: List of BED-format entries extracted from the GTF.
    """
    bed_records = []
    primary_chroms = set([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrMT"])
    with open(gtf_file, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if fields[2] != "gene":
                continue
            chromosome = fields[0]
            if not chromosome.startswith("chr"):
                chromosome = "chr" + chromosome
            if chromosome not in primary_chroms:
                continue
            start = int(fields[3]) - 1  # BED starts at 0
            end = int(fields[4])
            attributes = fields[8]
            gene_name = "N/A"
            for attribute in attributes.split(";"):
                if "gene_name" in attribute:
                    gene_name = attribute.split('"')[1]
                    break
            bed_records.append((chromosome, start, end, gene_name))

    df = pd.DataFrame(bed_records, columns=["chromosome", "start", "end", "gene_name"])
    df.to_csv("all_genes.bed", sep="\t", header=False, index=False)
    return bed_records


def intersect_snps(ranked_clusters, clusters_and_score):
    """
    Identifies top SNPs by intersecting ranked clusters with cluster scores.

    Parameters:
        ranked_clusters (list): Ordered list of cluster IDs.
        clusters_and_score (dict): Mapping of cluster IDs to SNP lists and scores.

    Returns:
        dict: Mapping of SNPs to their cluster rank and importance.
    """
    snp_clusters = BedTool("snp_clusters.bed")
    genes = BedTool("all_genes.bed")
    intersected = snp_clusters.intersect(genes, wa=True, wb=True)
    results = []
    for interval in intersected:
        fields = list(interval.fields)
        cluster_id = fields[4]
        score = clusters_and_score.get(cluster_id, 0)
        fields.append(str(score))
        results.append((cluster_id, fields))
    cluster_order = {cid: i for i, cid in enumerate(ranked_clusters)}
    results.sort(key=lambda x: cluster_order.get(x[0], float("inf")))
    header = [
        "snp_chr", "snp_start", "snp_end", "rsID", "cluster_id",
        "gene_chr", "gene_start", "gene_end", "gene_name",
        "cluster_imp_score"
    ]
    with open("cluster_gene_overlaps.tsv", "w") as out:
        out.write("\t".join(header) + "\n")
        for _, fields in results:
            out.write("\t".join(fields) + "\n")
    return results


def export_snp_summary(features_and_scores):
    """
    Exports SNP feature importance scores to a summary file.

    Parameters:
        features_and_scores (dict): Mapping of SNP IDs to their importance scores.

    Creates:
        'snp_summary.txt' file with SNP scores sorted by importance.
    """
    snps = BedTool("important_snps.bed")
    header = ["snp_chr", "snp_start", "snp_end", "rsID", "snp_score"]
    rows = []
    for interval in snps:
        fields = list(interval.fields)
        rsid = fields[3]
        score = features_and_scores.get(rsid, 0)
        fields.append(str(score))
        rows.append((score, fields))
    rows.sort(key=lambda x: x[0], reverse=True)
    with open("snp_summary.tsv", "w") as out:
        out.write("\t".join(header) + "\n")
        for _, fields in rows:
            out.write("\t".join(fields) + "\n")
    return [fields for _, fields in rows]


def export_cluster_summary(clusters_and_score):
    """
    Exports SNP cluster summary data to a text file.

    Parameters:
        clusters_and_score (dict): Mapping of clusters to their importance scores and SNPs.

    Creates:
        'cluster_summary.txt' file summarizing each cluster and its key SNPs.
    """
    snp_clusters = BedTool("snp_clusters.bed")
    cluster_data = defaultdict(lambda: {
        "rsIDs": [],
        "starts": [],
        "ends": [],
        "chrom": None
    })
    for interval in snp_clusters:
        fields = list(interval.fields)
        chrom = fields[0]
        start = fields[1]
        end = fields[2]
        rsid = fields[3]
        cluster_id = fields[4]
        data = cluster_data[cluster_id]
        data["chrom"] = chrom
        data["rsIDs"].append(rsid)
        data["starts"].append(start)
        data["ends"].append(end)
    rows = []
    for cluster_id, data in cluster_data.items():
        score = clusters_and_score.get(cluster_id, 0)
        rows.append((
            float(score),
            cluster_id,
            f"{data['chrom']}",
            ",".join(data["rsIDs"]),
            ",".join(data["starts"]),
            ",".join(data["ends"]),
            str(score)
        ))
    rows.sort(reverse=True, key=lambda x: x[0])
    header = ["cluster_id", "chromosome", "snps", "starts", "ends", "cluster_score"]
    with open("cluster_summary.tsv", "w") as out:
        out.write("\t".join(header) + "\n")
        for _, cluster_id, chrom, rsids, starts, ends, score in rows:
            out.write("\t".join([cluster_id, chrom, rsids, starts, ends, score]) + "\n")
    return rows
