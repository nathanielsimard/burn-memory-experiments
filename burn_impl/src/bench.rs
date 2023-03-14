pub fn bench<F>(func: F)
where
    F: FnOnce(),
{
    let start = std::time::Instant::now();
    func();
    let end = std::time::Instant::now();
    println!("Took {:?}", end - start);
}
