#=
VAE Dataset Utilities for Julia/Flux
=====================================

Utilities for loading and working with VAEDecodedImages-SDXL dataset
in Julia with Flux.jl for training VAE artifact detection models.

Usage:
    include("julia_utils.jl")
    using .VAEDatasetUtils

    dataset = VAEDetectorDataset("splits/train.txt", "manifest.csv", "/path/to/dataset")
    x, y = dataset[1]  # Get first sample
=#

module VAEDatasetUtils

using SHA
using CSV
using DataFrames
using FileIO
using Images
using Random
using Statistics

export VAEDetectorDataset, load_split, verify_checksum, compute_shake256
export load_manifest, get_image_pair, DataLoader, accuracy, train_epoch!

# SHAKE256 with d=256 (32 bytes output)
# Julia's SHA library uses extendable output functions
function compute_shake256(data::Vector{UInt8})::Vector{UInt8}
    ctx = SHA.SHA3_256_CTX()
    SHA.update!(ctx, data)
    return SHA.digest!(ctx)
end

function compute_shake256(filepath::AbstractString)::Vector{UInt8}
    data = read(filepath)
    return compute_shake256(data)
end

function bytes_to_hex(bytes::Vector{UInt8})::String
    return join(string(b, base=16, pad=2) for b in bytes)
end

function verify_checksum(filepath::AbstractString, expected_hex::AbstractString)::Bool
    if expected_hex == "skipped"
        return true
    end
    actual = bytes_to_hex(compute_shake256(filepath))
    return actual == expected_hex
end

# Load manifest CSV
function load_manifest(manifest_path::AbstractString)::DataFrame
    return CSV.read(manifest_path, DataFrame)
end

# Load split file (list of image IDs)
function load_split(split_path::AbstractString)::Vector{String}
    return filter(!isempty, readlines(split_path))
end

# Image pair structure
struct ImagePair
    id::String
    original_path::String
    vae_path::String
    original_size::Int64
    vae_size::Int64
    original_checksum::String
    vae_checksum::String
    stratum::Int
end

# Dataset for VAE artifact detection
struct VAEDetectorDataset
    pairs::Vector{ImagePair}
    base_path::String
    transform::Function
    verify_checksums::Bool
end

function VAEDetectorDataset(
    split_path::AbstractString,
    manifest_path::AbstractString,
    base_path::AbstractString;
    transform::Function = identity,
    verify_checksums::Bool = false
)
    # Load manifest and split
    manifest = load_manifest(manifest_path)
    split_ids = Set(load_split(split_path))

    # Filter to split
    pairs = ImagePair[]
    for row in eachrow(manifest)
        if row.id in split_ids
            push!(pairs, ImagePair(
                row.id,
                row.original_path,
                row.vae_path,
                row.original_size,
                row.vae_size,
                row.original_shake256_d256,
                row.vae_shake256_d256,
                row.stratum
            ))
        end
    end

    return VAEDetectorDataset(pairs, base_path, transform, verify_checksums)
end

Base.length(d::VAEDetectorDataset) = length(d.pairs)

function Base.getindex(d::VAEDetectorDataset, i::Int)
    pair = d.pairs[i]

    # Load images
    orig_path = joinpath(d.base_path, pair.original_path)
    vae_path = joinpath(d.base_path, pair.vae_path)

    # Verify checksums if enabled
    if d.verify_checksums
        if !verify_checksum(orig_path, pair.original_checksum)
            error("Checksum mismatch for original: $(pair.id)")
        end
        if !verify_checksum(vae_path, pair.vae_checksum)
            error("Checksum mismatch for VAE: $(pair.id)")
        end
    end

    # Load and convert to Float32 arrays
    orig_img = Float32.(channelview(load(orig_path)))
    vae_img = Float32.(channelview(load(vae_path)))

    # Apply transform
    orig_img = d.transform(orig_img)
    vae_img = d.transform(vae_img)

    # Return (original, label=0) and (vae, label=1)
    # For binary classification: 0 = original, 1 = VAE-processed
    return (orig_img, vae_img)
end

# Simple data loader for batching
struct DataLoader
    dataset::VAEDetectorDataset
    batch_size::Int
    shuffle::Bool
    indices::Vector{Int}
end

function DataLoader(dataset::VAEDetectorDataset; batch_size::Int=32, shuffle::Bool=true)
    indices = collect(1:length(dataset))
    if shuffle
        shuffle!(indices)
    end
    return DataLoader(dataset, batch_size, shuffle, indices)
end

function Base.iterate(dl::DataLoader, state::Int=1)
    if state > length(dl.indices)
        return nothing
    end

    batch_end = min(state + dl.batch_size - 1, length(dl.indices))
    batch_indices = dl.indices[state:batch_end]

    # Collect batch
    orig_batch = []
    vae_batch = []
    for i in batch_indices
        orig, vae = dl.dataset[i]
        push!(orig_batch, orig)
        push!(vae_batch, vae)
    end

    # Stack into arrays - combine originals and VAEs with labels
    # x: images, y: labels (0 for original, 1 for VAE)
    n = length(batch_indices)
    x = cat(orig_batch..., vae_batch..., dims=4)
    y = vcat(zeros(Float32, n), ones(Float32, n))

    return ((x, y), batch_end + 1)
end

Base.length(dl::DataLoader) = ceil(Int, length(dl.dataset) / dl.batch_size)

# Training utilities
function accuracy(model, x, y)
    predictions = model(x)
    pred_labels = predictions .> 0.5f0
    return mean(pred_labels .== y)
end

# Example training loop (requires Flux.jl)
function train_epoch!(model, opt_state, train_loader, loss_fn)
    total_loss = 0.0f0
    n_batches = 0

    for (x, y) in train_loader
        # Compute gradients and update
        grads = Flux.gradient(model) do m
            loss_fn(m(x), y)
        end
        Flux.update!(opt_state, model, grads[1])

        total_loss += loss_fn(model(x), y)
        n_batches += 1
    end

    return total_loss / n_batches
end

# Comparison utilities for random vs stratified splits
function compare_splits(
    manifest_path::AbstractString,
    random_split_dir::AbstractString,
    stratified_split_dir::AbstractString
)
    manifest = load_manifest(manifest_path)

    results = Dict{String, Dict{String, Any}}()

    for split_name in ["train", "test", "val", "calibration"]
        random_ids = Set(load_split(joinpath(random_split_dir, "random_$(split_name).txt")))
        strat_ids = Set(load_split(joinpath(stratified_split_dir, "stratified_$(split_name).txt")))

        # Calculate stratum distribution
        random_strata = [row.stratum for row in eachrow(manifest) if row.id in random_ids]
        strat_strata = [row.stratum for row in eachrow(manifest) if row.id in strat_ids]

        results[split_name] = Dict(
            "random_count" => length(random_ids),
            "stratified_count" => length(strat_ids),
            "random_stratum_dist" => countmap(random_strata),
            "stratified_stratum_dist" => countmap(strat_strata),
            "overlap" => length(intersect(random_ids, strat_ids))
        )
    end

    return results
end

# Simple countmap implementation
function countmap(xs)
    counts = Dict{Any, Int}()
    for x in xs
        counts[x] = get(counts, x, 0) + 1
    end
    return counts
end

end # module

#=
Example usage with Flux.jl:

using Flux
include("julia_utils.jl")
using .VAEDatasetUtils

# Load dataset
train_data = VAEDetectorDataset(
    "splits/random_train.txt",
    "manifest.csv",
    "/path/to/vae-dataset"
)

# Create data loader
train_loader = DataLoader(train_data, batch_size=32, shuffle=true)

# Simple binary classifier
model = Chain(
    Conv((3, 3), 3 => 32, relu),
    MaxPool((2, 2)),
    Conv((3, 3), 32 => 64, relu),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(64 * 190 * 190, 128, relu),
    Dense(128, 1, sigmoid)
)

# Training
opt = Flux.setup(Adam(1e-4), model)
loss_fn = Flux.binarycrossentropy

for epoch in 1:10
    loss = train_epoch!(model, opt, train_loader, loss_fn)
    println("Epoch $epoch: loss = $loss")
end
=#
