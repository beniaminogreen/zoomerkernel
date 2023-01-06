test_that("RPC basis provides good approximation of kernel of volcano matrix", {
    bw <- .000005

    target <- rbf_kernel_matrix(volcano, bw)

    expect_lt(norm(target - crossprod(t(rpchol(volcano, 5, bw))), "F"),2)
    expect_lt(norm(target - crossprod(t(rpchol(volcano, 10, bw))), "F"),.3)
    expect_lt(norm(target - crossprod(t(rpchol(volcano, 20, bw))), "F"),.05)
})
