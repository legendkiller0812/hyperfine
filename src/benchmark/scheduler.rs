use colored::*;

use super::benchmark_result::BenchmarkResult;
use super::executor::{Executor, MockExecutor, RawExecutor, ShellExecutor};
use super::{relative_speed, Benchmark};
use std::{thread, time};

use crate::command::Commands;
use crate::export::ExportManager;
use crate::options::{ExecutorKind, Options, OutputStyleOption};
use crate::output::progress_bar::get_progress_bar;
use anyhow::Result;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::sync::Arc;
use std::sync::Mutex;

pub struct Scheduler<'a> {
    commands: &'a Commands<'a>,
    options: &'a Options,
    export_manager: &'a ExportManager,
    results: Vec<BenchmarkResult>,
}

impl<'a> Scheduler<'a> {
    pub fn new(
        commands: &'a Commands,
        options: &'a Options,
        export_manager: &'a ExportManager,
    ) -> Self {
        Self {
            commands,
            options,
            export_manager,
            results: vec![],
        }
    }

    pub fn run_benchmarks(&mut self) -> Result<()> {
        let m = MultiProgress::new();

        let mut pb = std::iter::repeat_with(|| {
            let progress_bar = if self.options.output_style != OutputStyleOption::Disabled {
                Some(get_progress_bar(
                    20,
                    &format!("Performing batch overhead {:?}/{:?}", 0, 20),
                    self.options.output_style,
                ))
            } else {
                None
            };
            m.add(progress_bar.unwrap())
        })
        .take(self.commands.num_commands())
        .collect::<Vec<_>>();

        m.println("Starting the benchmark. Please hold...").unwrap();
        let self_clone = self.options.clone();
        let commands_clone = self.commands.clone();
        let mut localexecutor: Box<dyn Executor> = match self_clone.executor_kind {
            ExecutorKind::Raw => Box::new(RawExecutor::new(self_clone)),
            ExecutorKind::Mock(ref shell) => Box::new(MockExecutor::new(shell.clone())),
            ExecutorKind::Shell(ref shell) => {
                Box::new(ShellExecutor::new(shell, self_clone))
            }
        };

        localexecutor.calibrate(pb.get(0).unwrap())?;
        let localexecutor:Arc<dyn Executor> = Arc::from(localexecutor);


        let pool =  rayon::ThreadPoolBuilder::new()
        .num_threads(self.options.num_threads as usize)
        .build()
        .unwrap();
        let results_vec = Arc::new(Mutex::new(Vec::new()));
        pool.scope(|s| {
            for (number, cmd) in commands_clone.iter().enumerate() {
                let t_executor = localexecutor.clone();
                let t_pbar = Some(pb.get(number).unwrap()).clone();
                let t_multiprogress = m.clone();
                let t_cmd = cmd.clone();
                let t_options = self_clone.clone();
                let t_results = results_vec.clone();


                let handle = s.spawn(move |_| {
                    let ret = Benchmark::new(
                        number,
                        &t_cmd,
                        t_options,
                        t_executor.to_owned().as_ref() ,
                        t_pbar,
                        &t_multiprogress,
                    )
                    .run();
                    match ret {
                        Ok(res) => {
                            t_results.lock().unwrap().push(res);
                        }
                        Err(status) => panic!("Problem in thread {:?}: {:?}", number, status),
                    };
                });

            }
        });
        self.results = results_vec.lock().unwrap().to_owned();

        // We export (all results so far) after each individual benchmark, because
        // we would risk losing all results if a later benchmark fails.
        self.export_manager
            .write_results(&self.results, self.options.time_unit)?;
        Ok(())
    }

    pub fn print_relative_speed_comparison(&self) {
        if self.options.output_style == OutputStyleOption::Disabled {
            return;
        }

        if self.results.len() < 2 {
            return;
        }

        if let Some(mut annotated_results) = relative_speed::compute(&self.results) {
            annotated_results.sort_by(|l, r| relative_speed::compare_mean_time(l.result, r.result));

            let fastest = &annotated_results[0];
            let others = &annotated_results[1..];

            println!("{}", "Summary".bold());
            println!("  '{}' ran", fastest.result.command.cyan());

            for item in others {
                println!(
                    "{}{} times faster than '{}'",
                    format!("{:8.2}", item.relative_speed).bold().green(),
                    if let Some(stddev) = item.relative_speed_stddev {
                        format!(" ± {}", format!("{:.2}", stddev).green())
                    } else {
                        "".into()
                    },
                    &item.result.command.magenta()
                );
            }
        } else {
            eprintln!(
                "{}: The benchmark comparison could not be computed as some benchmark times are zero. \
                 This could be caused by background interference during the initial calibration phase \
                 of hyperfine, in combination with very fast commands (faster than a few milliseconds). \
                 Try to re-run the benchmark on a quiet system. If it does not help, you command is \
                 most likely too fast to be accurately benchmarked by hyperfine.",
                 "Note".bold().red()
            );
        }
    }
}
