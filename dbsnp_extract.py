import pysam
import pickle

# VCF file and index
vcf_file = "/Volumes/seagate/GCF_000001405.25.gz"
vcf = pysam.VariantFile(vcf_file)

# Mapping of chromosome labels to contigs in GRCh37
CONTIG_MAP = {
    "1": "NC_000001.10", "2": "NC_000002.11", "3": "NC_000003.11",
    "4": "NC_000004.11", "5": "NC_000005.9", "6": "NC_000006.11",
    "7": "NC_000007.13", "8": "NC_000008.10", "9": "NC_000009.11",
    "10": "NC_000010.10", "11": "NC_000011.9", "12": "NC_000012.11",
    "13": "NC_000013.10", "14": "NC_000014.8", "15": "NC_000015.9",
    "16": "NC_000016.9", "17": "NC_000017.10", "18": "NC_000018.9",
    "19": "NC_000019.9", "20": "NC_000020.10", "21": "NC_000021.8",
    "22": "NC_000022.10", "X": "NC_000023.10", "Y": "NC_000024.9",
    "MT": "NC_012920.1"
}


def generate_snp_dict(filename):
    """Generates a SNP dictionary from an Illumina manifest file."""
    snp_dict = {}
    ind = 0
    with open(filename) as f:
        for line in f:
            if ind <= 7:  # Skip the first 8 header lines
                ind += 1
                continue
            lst = line.strip().split(",")
            if lst[0] == "[Controls]":
                break  # End of SNP entries
            if lst[9] == "0":
                continue  # Skip entries with chromosome 0 (non-mapped)
            snp_dict[lst[1]] = (lst[9], lst[10])
    print(f"Loaded {len(snp_dict)} SNPs into snp_dict.")
    with open("snp_dict.pkl", "wb") as f:
        pickle.dump(snp_dict, f)
    return snp_dict


def fetch_snps(snp_dict):
    """Fetches SNP rsIDs from VCF for given chromosome and position, supporting XY lookup."""
    snp_ref = {}
    count_matches = 0
    count_not_found = 0
    for snp_name, (chromosome, pos_str) in snp_dict.items():
        position = int(pos_str)
        if chromosome == "XY":
            chrom_options = ["X", "Y"]
        else:
            chrom_options = [chromosome]
        found = False
        for chrom in chrom_options:
            contig_name = CONTIG_MAP.get(chrom)
            if contig_name is None:
                print(f"Chromosome {chrom} not found in contig map. Skipping {snp_name}...")
                continue
            try:
                records = list(vcf.fetch(contig_name, position - 1, position))
                if not records:
                    print(f"No match found on {chrom} for {snp_name} at {chrom}:{position}")
                    continue
                for record in records:
                    found = True
                    count_matches += 1
                    snp_ref[snp_name] = {
                        "chromosome": chrom,
                        "position": position,
                        "rsID": record.id
                    }
                    print(f"{snp_name}: Found {record.id} on {chrom} at {position}")
                if found:
                    break
            except ValueError as e:
                print(f"❗ Fetch failed for {chrom}:{position} - {e}")
        if not found:
            count_not_found += 1
            snp_ref[snp_name] = None
            print(f"❓ No match found for {snp_name} after checking {chrom_options}")
    print(f"\nFinished! Found {count_matches} SNPs. {count_not_found} SNPs were not found.\n")
    with open("snp_ref.pkl", "wb") as f:
        pickle.dump(snp_ref, f)
    return snp_ref

