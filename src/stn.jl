using Images
using ImageView
using Flux#, Flux.Data.MNIST
using Flux: onehotbatch, onecold
using Flux.Tracker: update!
using LinearAlgebra
using Base.Iterators: partition
using BatchedRoutines
using MAT

BATCH_SIZE = 128
# imgs = MNIST.images()
# labels = MNIST.labels()
#
# data = [reshape(hcat(Array(channelview.(imgs))...), 28, 28, 1,:) for imgs in partition(imgs, BATCH_SIZE)]
# data = gpu.(data)

data_file = matopen("C:/Users/manju/Downloads/training_and_validation_batches/1.mat")
affNIST_data = read(data_file, "affNISTdata")
imgs = affNIST_data["image"]
labels = affNIST_data["label_one_of_n"]

train_data = []
for i in 1:60000
	push!(train_data, permutedims(reshape(imgs[:, i], 40, 40, 1), [2, 1, 3]))
end
train_data = reshape(cat(train_data..., dims = 4), 40, 40, 1, 60000)


NUM_EPOCHS = 50
training_steps = 0
num_classes = 10

function affine_grid_generator(height, width, theta)
	batch_size = size(theta)[3]
	x = LinRange(-1, 1, width)
	y = LinRange(-1, 1, height)
	x_t_flat = reshape(repeat(x, height), 1, height*width)
	y_t_flat = reshape(repeat(transpose(y), width), 1, height*width)
	all_ones = ones(eltype(x_t_flat), 1, size(x_t_flat)[2])

	sampling_grid = vcat(x_t_flat, y_t_flat, all_ones)
	sampling_grid = Array(reshape(transpose(repeat(transpose(sampling_grid), batch_size)), 3, size(x_t_flat)[2], batch_size))

	batch_grids = batched_gemm('N', 'N', theta, sampling_grid)
	y_s = reshape(batch_grids[2, :, :], width, height, batch_size)
	x_s = reshape(batch_grids[1, :, :], width, height, batch_size)
	return x_s, y_s
end

function get_pixel_values(img, x, y)
	batch_size = size(x, 3)
	width = size(img, 1)
	height = size(img, 2)
	channels = size(img, 3)

	x = trunc.(Int, x)
	y = trunc.(Int, y)
	x_indices = []
	y_indices = []
	for i in 1:batch_size
		push!(x_indices, diag(x[:, :, i]))
		push!(y_indices, diag(y[:, :, i]))
	end
	x_indices = reshape(cat(x_indices..., dims = 4), width, batch_size)
	y_indices = reshape(cat(y_indices..., dims = 4), height, batch_size)
	batch = []
	#println(size(img))
	println(batch_size)
	for i in 1:batch_size
		pic = colorview(RGB, reshape(img[:, :, :, i], channels, width, height))
		new_pic = pic[y_indices[:, i], x_indices[:, i]]
		# new_pic = pic
		# for j in 1:height
		# 	for k in 1:width
		# 		new_pic[j, k] = pic[y[k, j, i], x[k, j, i]]
		# 	end
		# end
		push!(batch, reshape(Float64.(channelview(new_pic)), width, height, channels))
	end
	batch = reshape(cat(batch..., dims = 4),width, height, channels, batch_size)
	return batch
end

function bilinear_sampler(img, x, y)
	height = size(img)[1]
	width = size(img)[2]
	channels = size(img)[3]
	batch_size = size(img)[4]
	max_y = height
	max_x = width

	x = 0.5*(x .+ 1.0)*(max_x)
	y = 0.5*(y .+ 1.0)*(max_y)

	x0 = trunc.(Int, x)
	x1 = x0 .+ 1.0
	y0 = trunc.(Int, y)
	y1 = y0 .+ 1.0
	x0 = clamp.(x0, 1, max_x)
	x1 = clamp.(x1, 1, max_x)
	y0 = clamp.(y0, 1, max_y)
	y1 = clamp.(y1, 1, max_y)

	w1 = (x1 - x).*(y1 - y)
	w2 = (x1 - x).*(y - y0)
	w3 = (x - x0).*(y1 - y)
	w4 = (x - x0).*(y - y0)

	valA = get_pixel_values(img, x0, y0)
	valB = get_pixel_values(img, x0, y1)
	valC = get_pixel_values(img, x1, y0)
	valD = get_pixel_values(img, x1, y1)

	# new_img = colorview(RGB, reshape(valA[:, :, :, 1], 3, 64, 64))
	# imshow(new_img)
	# new_img = colorview(RGB, reshape(valB[:, :, :, 1], 3, 64, 64))
	# imshow(new_img)
	# new_img = colorview(RGB, reshape(valC[:, :, :, 1], 3, 64, 64))
	# imshow(new_img)
	# new_img = colorview(RGB, reshape(valD[:, :, :, 1], 3, 64, 64))
	# imshow(new_img)

	weight1 = []
	weight2 = []
	weight3 = []
	weight4 = []
	for i in 1:channels
		push!(weight1, w1)
		push!(weight2, w2)
		push!(weight3, w3)
		push!(weight4, w4)
	end

	weight1 = permutedims(reshape(cat(weight1..., dims = 4), height, width, batch_size, channels), [1, 2, 4, 3])
	weight2 = permutedims(reshape(cat(weight2..., dims = 4), height, width, batch_size, channels), [1, 2, 4, 3])
	weight3 = permutedims(reshape(cat(weight3..., dims = 4), height, width, batch_size, channels), [1, 2, 4, 3])
	weight4 = permutedims(reshape(cat(weight4..., dims = 4), height, width, batch_size, channels), [1, 2, 4, 3])

	resultant = weight1 .* valA + weight2 .* valB + weight3 .* valC + weight4 .* valD
	return valD
end

localization_net = Chain(MaxPool((2, 2)), Conv((5, 5), 1 => 20, stride = (1, 1), pad = (0, 0)),
					MaxPool((2, 2)), Conv((5, 5), 20 => 20, stride = (1, 1), pad = (0, 0)),
					x -> reshape(x, :, size(x, 4)),
					Dense(1620, 50), x -> relu.(x),
					Dense(50, 4))

function transformer(input_batch, loc_net_output)
	width = size(input_batch)[1]
	height = size(input_batch)[2]
	channels = size(input_batch)[3]
	batch_size = size(input_batch)[4]
	scale_factor = loc_net_output[1, :]
	trans_width = loc_net_output[2, :]
	trans_height = loc_net_output[3, :]
	angle = loc_net_output[4, :]
	theta = Matrix{Float64}(I, 2, 3)
	theta = Array(reshape(transpose(repeat(transpose(theta), batch_size)), 2, 3, batch_size))
	for i in 1:batch_size
		theta[:, :, i] = scale_factor[i]*theta[:, :, i]
		theta[1, 3, i] = trans_width[i]
		theta[2, 3, i] = trans_height[i]
	end
	out = bilinear_sampler(input_batch, affine_grid_generator(height, width, theta)...)
	output_batch = []
	for i in 1:batch_size
		pic = colorview(RGB, reshape(out[:, :, :, i], channels, width, height))
		rotated_pic = imrotate(pic, angle[i], axes(pic))
		push!(output_batch, reshape(Float64.(channelview(rotated_pic)), width, height, channels))
	end
	output_batch = reshape(cat(output_batch..., dims = 4), width, height, channels, batch_size)
	return output_batch
end

model = Chain(Conv((3, 3), 3 => 32, relu ), MaxPool((2, 2)),
			Conv((3, 3), 32 => 32, relu), MaxPool((2, 2)),
			x -> reshape(x, :, size(x, 4)),
			Dense(704, 256),
			x -> relu.(x),
			Dense(256, num_classes),
			softmax)

# function training(X)
# 	loc_net = localization_net(X)
# 	transformed_X = transformer(X, loc_net)
# 	output = model(transformed_X)
#
# end


for epoch in 1:NUM_EPOCHS
	println("-------- Epoch : $epoch ---------")
	for X in data
		training(X)
	end
end
