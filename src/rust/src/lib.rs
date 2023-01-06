use extendr_api::prelude::*;
use indicatif::{ParallelProgressIterator, ProgressBar};

use rand::distributions::WeightedIndex;
use rand::prelude::*;

use std::cmp;

use rayon::prelude::*;

fn l2_norm(x: ArrayView1<f64>) -> f64 {
    x.map(|x| x.powi(2)).sum().sqrt()
}

fn rbf_kernel(x: ArrayView1<f64>, y: ArrayView1<f64>, bw: f64) -> f64 {
    x.iter()
        .zip(y)
        .map(|(x, y)| -1.0 * (x - y).abs().powi(2) * bw)
        .sum::<f64>()
        .exp()
}

/// Return string `"Hello world!"` to R.
/// @export
#[extendr]
fn rbf_kernel_matrix(x_robj: Robj, bw: f64) -> Robj {
    let x_mat = <ArrayView2<f64>>::from_robj(&x_robj).unwrap();

    let mut out_mat: Array2<f64> = Array2::zeros((x_mat.nrows(), x_mat.nrows()));

    for i in 0..x_mat.nrows() {
        for j in 0..x_mat.nrows() {
            let res = rbf_kernel(
                x_mat.index_axis(Axis(0), i),
                x_mat.index_axis(Axis(0), j),
                bw,
            );
            out_mat[[i, j]] = res;
            out_mat[[j, i]] = res;
        }
    }

    Robj::try_from(&out_mat).into()
}

fn rust_rpchol(x_mat: ArrayView2<f64>, k: usize, bw: f64) -> Array2<f64> {
    let mut f_mat: Array2<f64> = Array2::zeros((x_mat.nrows(), k));

    let mut d = Array1::from_iter(x_mat.axis_iter(Axis(0)).map(|x| rbf_kernel(x, x, bw)));

    let mut rng = thread_rng();

    let k = cmp::min(k, x_mat.nrows());
    for i in 0..k {
        let dist = WeightedIndex::new(&d).unwrap();

        let s = dist.sample(&mut rng);

        let mut g: Array1<f64> = Array1::from_iter(
            x_mat
                .axis_iter(Axis(0))
                .map(|x| rbf_kernel(x.view(), x_mat.index_axis(Axis(0), s), bw)),
        );

        let corr = f_mat
            .slice(s![.., ..(i as usize)])
            .dot(&f_mat.slice(s![s, ..(i as usize)]).t());

        g = g - corr;

        g /= g[s].sqrt();

        f_mat.index_axis_mut(Axis(1), i as usize).assign(&g);

        d = d - (&g * &g);
        d = d.map(|d| if d < &0.0 { 0.0 } else { *d });

        if d.sum() < 0.0001 {
            break;
        };
    }

    f_mat
}

/// calculate the random-pivot cholesky factorization
/// @export
#[extendr]
fn rpchol(x_robj: Robj, k: i64, bw: f64) -> Robj {
    let x_mat = <ArrayView2<f64>>::from_robj(&x_robj).unwrap();

    let f_mat = rust_rpchol(x_mat, k as usize, bw);

    Robj::try_from(&f_mat).into()
}

fn balance_bootstrap(f_mat: ArrayView2<f64>, w: &[f64]) -> f64 {
    let mut rng = rand::thread_rng();

    let alternate_z =
        Array1::from_iter(w.iter().map(|w| if rng.gen_bool(*w) { 1.0 } else { -1.0 }));
    l2_norm(f_mat.dot(&f_mat.t().dot(&alternate_z)).view())
}

/// Kernel Balance Statistic
/// @export
#[extendr]
fn kbalstat(x_robj: Robj, z: &[f64], w: &[f64], k: i64, bw: f64, nboots: u64) -> List {
    let x_mat = <ArrayView2<f64>>::from_robj(&x_robj).unwrap();

    let f_mat = rust_rpchol(x_mat, k as usize, bw);

    let z = arr1(z);

    let pb = ProgressBar::new(nboots);

    let boots = (0..nboots)
        .into_par_iter()
        .progress_with(pb)
        .map(|_| balance_bootstrap(f_mat.view(), w))
        .collect::<Vec<f64>>();

    list!(
        statistic = l2_norm(f_mat.dot(&f_mat.t().dot(&z)).view()),
        boots = boots.iter().map(|i| Rfloat::from(*i)).collect::<Doubles>()
    )
}

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod zoomerkernel;
    fn rbf_kernel_matrix;
    fn rpchol;
    fn kbalstat;
}
