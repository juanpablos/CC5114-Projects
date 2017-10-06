import csv
import re


status_file = "scala_status.txt"
statistics_file = "scala_statistics.txt"
output_file = "scala_out_alt.csv"

data = {} # key:[list] - sha:data

def get_high_level_data(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        per_commit_temp = re.split(r'^commit|\ncommit', f.read())
        per_commit = [i for i in per_commit_temp if i is not None]

    for commit in reversed(per_commit[1:]):
        commit_lines = commit.split('\n')
        commit_sha = commit_lines[0].strip()

        deleted_files = 0
        new_files = 0
        modified_files = 0
        for line in commit_lines:
            if len(line.strip()) > 2:
                if line[1] == '\t':
                    action = line[0]
                    if action == 'D':
                        deleted_files += 1
                    if action == 'A':
                        new_files += 1
                    if action == 'M':
                        modified_files += 1

        data[commit_sha] = [new_files, deleted_files, modified_files]


def get_statistics(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        per_commit_temp = re.split(r'^commit|\ncommit', f.read())
        per_commit = [i for i in per_commit_temp if i is not None]

    for commit in reversed(per_commit[1:]):
        commit_lines = commit.split('\n')
        commit_sha = commit_lines[0].strip()

        inserted_lines = 0
        deleted_lines = 0

        for line in commit_lines:
            if ('insertions(+)' in line) or ('deletions(-)' in line):
                temp = line.split(', ')
                if len(temp) == 3:
                    inserted_lines = int(temp[1].split(' ')[0].strip())
                    deleted_lines = int(temp[2].split(' ')[0].strip())
                else:
                    if 'insertions' in temp[1]:
                        inserted_lines = int(temp[1].split(' ')[0].strip())
                    else:
                        deleted_lines = int(temp[1].split(' ')[0].strip())

        data[commit_sha] += [inserted_lines, deleted_lines]


def out_to_file_trad(outfile):
    with open(outfile, 'w', newline="\n") as o:
        out = csv.writer(o)
        out.writerow(['new_files', 'deleted_files', 'modified_files', 'inserted_lines', 'deleted_lines',
                      'new_files_pred', 'deleted_files_pred', 'modified_files_pred', 'inserted_lines_pred',
                      'deleted_lines_pred'])

        values = list(data.values())
        for i in range(1, len(values)):
            out.writerow(values[i-1] + values[i])

def out_to_file_alternative(outfile):
    with open(outfile, 'w', newline="\n") as o:
        out = csv.writer(o)
        out.writerow(['new_files', 'deleted_files', 'modified_files', 'inserted_lines', 'deleted_lines',
                      'modified_files_pred', 'inserted_lines_pred', 'deleted_lines_pred'])

        values = list(data.values())
        for i in range(1, len(values)):
            out.writerow(values[i-1] + values[i][2:])


if __name__ == '__main__':
    get_high_level_data(status_file)
    get_statistics(statistics_file)
    #out_to_file_trad(output_file)
    out_to_file_alternative(output_file)
