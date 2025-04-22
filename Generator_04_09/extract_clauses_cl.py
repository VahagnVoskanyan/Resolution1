import json

# Load the JSON data from _rs.jsonl
with open('Res_Pairs/gen_problem_CAT001-0_N=10_T=5.0_0_rs.jsonl', 'r') as f:
    json_data = [json.loads(line) for line in f]

# Extract clauses from JSON data
clauses_data = json_data[0]['clauses']  # First line contains all clauses
clauses_dict = {clause[0]: {'type': clause[1], 'literals': clause[2]} for clause in clauses_data}

# Parse the solved.txt file to extract the clauses used in the proof
proof_clauses = []
with open('Output/gen_problem_CAT001-0_N=10_T=5.0_0_solved.txt', 'r') as f:
    for line in f:
        if line.startswith('fof('):
            # Extract clause ID
            clause_id = line.split(',')[0].split('(')[1]
            proof_clauses.append(clause_id)

# # Map proof clauses to original clauses
for clause_id in proof_clauses:
    # Handle mapping from f12 to u12, etc. if needed
    if clause_id.startswith('f'):
        original_id = 'u' + clause_id[1:]  # This is a simplification; actual mapping may vary
        if original_id in clauses_dict:
            print(f"Found proof clause {clause_id} as {original_id} in original problem")
            print(clauses_dict[original_id])