pub mod benchmark_result;
pub mod executor;
pub mod relative_speed;
pub mod scheduler;
pub mod timing_result;
use crate::benchmark::executor::MockExecutor;
use crate::benchmark::executor::RawExecutor;
use crate::benchmark::executor::ShellExecutor;
use crate::command::Command;
use crate::options::{CmdFailureAction, ExecutorKind, Options, OutputStyleOption};
use crate::outlier_detection::{modified_zscores, OUTLIER_THRESHOLD};
use crate::output::format::{format_duration, format_duration_unit};
use crate::output::warnings::Warnings;
use crate::parameter::ParameterNameAndValue;
use crate::util::exit_code::extract_exit_code;
use crate::util::min_max::{max, min};
use crate::util::units::Second;

use indicatif::MultiProgress;
use indicatif::ProgressBar;
use rayon::prelude::*;

use benchmark_result::BenchmarkResult;
use std::cmp;
use std::ops::Deref;
use std::process::ExitStatus;
use std::sync::Arc;
use std::sync::Mutex;

use timing_result::TimingResult;

use anyhow::{anyhow, Result};
use colored::*;
use statistical::{mean, median, standard_deviation};

use self::executor::Executor;

/// Threshold for warning about fast execution time
pub const MIN_EXECUTION_TIME: Second = 5e-3;

pub struct Benchmark<'a> {
    number: usize,
    command: &'a Command<'a>,
    options: &'a Options,
    executor: &'a dyn Executor,
    progress_bar: std::option::Option<&'a ProgressBar>,
    multiprogress: &'a MultiProgress,
}

impl<'a> Benchmark<'a> {
    pub fn new(
        number: usize,
        command: &'a Command<'a>,
        options: &'a Options,
        executor: &'a dyn Executor,
        progress_bar: std::option::Option<&'a ProgressBar>,
        multiprogress: &'a MultiProgress,
    ) -> Self {
        Benchmark {
            number,
            command,
            options,
            executor,
            progress_bar,
            multiprogress,
        }
    }
    /// Run setup, cleanup, or preparation commands
    fn run_intermediate_command(
        &self,
        command: &Command<'_>,
        error_output: &'static str,
    ) -> Result<TimingResult> {
        self.executor
            .run_command_and_measure(command, Some(CmdFailureAction::RaiseError))
            .map(|r| r.0)
            .map_err(|_| anyhow!(error_output))
    }

    /// Run the command specified by `--setup`.
    fn run_setup_command(
        &self,
        parameters: impl IntoIterator<Item = ParameterNameAndValue<'a>>,
    ) -> Result<TimingResult> {
        let command = self
            .options
            .setup_command
            .as_ref()
            .map(|setup_command| Command::new_parametrized(None, setup_command, parameters));

        let error_output = "The setup command terminated with a non-zero exit code. \
                            Append ' || true' to the command if you are sure that this can be ignored.";

        Ok(command
            .map(|cmd| self.run_intermediate_command(&cmd, error_output))
            .transpose()?
            .unwrap_or_default())
    }

    /// Run the command specified by `--cleanup`.
    fn run_cleanup_command(
        &self,
        parameters: impl IntoIterator<Item = ParameterNameAndValue<'a>>,
    ) -> Result<TimingResult> {
        let command = self
            .options
            .cleanup_command
            .as_ref()
            .map(|cleanup_command| Command::new_parametrized(None, cleanup_command, parameters));

        let error_output = "The cleanup command terminated with a non-zero exit code. \
                            Append ' || true' to the command if you are sure that this can be ignored.";

        Ok(command
            .map(|cmd| self.run_intermediate_command(&cmd, error_output))
            .transpose()?
            .unwrap_or_default())
    }

    /// Run the command specified by `--prepare`.
    fn run_preparation_command(&self, command: &Command<'_>) -> Result<TimingResult> {
        let error_output = "The preparation command terminated with a non-zero exit code. \
                            Append ' || true' to the command if you are sure that this can be ignored.";

        self.run_intermediate_command(command, error_output)
    }

    /// Run the benchmark for a single command
    pub fn run(&self) -> Result<BenchmarkResult> {
        let command_name = self.command.get_name();

        let mut pool = rayon::ThreadPoolBuilder::new()
        .num_threads(10)
        .build()
        .unwrap();
        let mut batch_average: f32 = 0.0;
        //Finding Batch overhead
        {
            let batch_runs = 10;
            let mut times_batch: Vec<Second> = vec![];
            let mutex_times_batch: Arc<Mutex<Vec<Second>>> =
                Arc::new(Mutex::new(times_batch.clone()));
            let command_name = String::from("batch");
            let command_expr = self.options.batch_cmd.as_ref().unwrap();
            let batch_command = Command::new(Some(&command_name), &command_expr);

            let local_command = Arc::new(batch_command.clone());

            let self_clone = self.clone();
            let localexecutor: Arc<dyn Executor> = match self_clone.options.executor_kind {
                ExecutorKind::Raw => Arc::new(RawExecutor::new(self_clone.options)),
                ExecutorKind::Mock(ref shell) => Arc::new(MockExecutor::new(shell.clone())),
                ExecutorKind::Shell(ref shell) => {
                    Arc::new(ShellExecutor::new(shell, self_clone.options))
                }
            };
            if let Some(bar) = self.progress_bar {
                bar.set_length(batch_runs);
                bar.set_message(format!(
                    "Performing batch overhead {:?}/{:?}",
                    0, batch_runs
                ));
            }
            pool.scope(|s| {
                let mut handles = Vec::new();
                for cnt in 0..batch_runs {
                    let t_times_batch = mutex_times_batch.clone();
                    let t_executor = localexecutor.clone();
                    let t_command = local_command.clone();
                    let t_bar = self.progress_bar.clone();
                    let handle = s.spawn(move |_| {
                        let mut success: bool = false;
                        let ret = t_executor.run_command_and_measure(&t_command, None);
                        match ret {
                            Ok(res) => {
                                t_times_batch.lock().unwrap().push(res.0.time_real);
                                success = res.1.success();
                            }
                            Err(status) => panic!("Problem in thread {:?}: {:?}", cnt, status),
                        };

                        if let Some(bar) = t_bar.as_ref() {
                            bar.inc(1);
                            bar.set_message(format!(
                                "{:?}. batch overhead calc {:?}/{:?}",
                                self.number,
                                bar.position(),
                                batch_runs
                            ));
                        }
                    });
                    handles.push(handle);
                }
            });
            if let Some(bar) = self.progress_bar {
                bar.reset()
            }
            times_batch = mutex_times_batch.lock().unwrap().clone();
            batch_average = times_batch.iter().sum::<f64>() as f32 / times_batch.len() as f32;
        }

        let mut times_real: Vec<Second> = vec![];
        let mut times_user: Vec<Second> = vec![];
        let mut times_system: Vec<Second> = vec![];
        let mut exit_codes: Vec<Option<i32>> = vec![];
        let mut all_succeeded = true;

        let preparation_command = self.options.preparation_command.as_ref().map(|values| {
            let preparation_command = if values.len() == 1 {
                &values[0]
            } else {
                &values[self.number]
            };
            Command::new_parametrized(
                None,
                preparation_command,
                self.command.get_parameters().iter().cloned(),
            )
        });
        let run_preparation_command = || {
            preparation_command
                .as_ref()
                .map(|cmd| self.run_preparation_command(cmd))
                .transpose()
        };

        self.run_setup_command(self.command.get_parameters().iter().cloned())?;

        // Warmup phase
        if self.options.warmup_count > 0 {
            if let Some(bar) = self.progress_bar {
                bar.set_length(self.options.warmup_count);
                bar.set_message("Performing warmup runs");
            }

            for _ in 0..self.options.warmup_count {
                let _ = run_preparation_command()?;
                let _ = self.executor.run_command_and_measure(self.command, None)?;
                if let Some(bar) = self.progress_bar.as_ref() {
                    bar.inc(1)
                }
            }
            if let Some(bar) = self.progress_bar.as_ref() {
                bar.reset()
            }
        }

        // Set up progress bar (and spinner for initial measurement)
        if let Some(bar) = self.progress_bar {
            bar.set_length(self.options.run_bounds.min);
            bar.set_message("Initial time measurement");
        }

        let preparation_result = run_preparation_command()?;
        let preparation_overhead =
            preparation_result.map_or(0.0, |res| res.time_real + self.executor.time_overhead());

        // Initial timing run
        let (res, status) = self.executor.run_command_and_measure(self.command, None)?;
        let success = status.success();

        // Determine number of benchmark runs
        let runs_in_min_time = (self.options.min_benchmarking_time
            / (res.time_real + self.executor.time_overhead() + preparation_overhead))
            as u64;

        let count = {
            let min = cmp::max(runs_in_min_time, self.options.run_bounds.min);

            self.options
                .run_bounds
                .max
                .as_ref()
                .map(|max| cmp::min(min, *max))
                .unwrap_or(min)
        };

        let count_remaining = count - 1;

        // Save the first result
        times_real.push(res.time_real);
        times_user.push(res.time_user);
        times_system.push(res.time_system);
        exit_codes.push(extract_exit_code(status));

        all_succeeded = all_succeeded && success;

        // Re-configure the progress bar
        if let Some(bar) = self.progress_bar.as_ref() {
            bar.set_length(count)
        }
        if let Some(bar) = self.progress_bar.as_ref() {
            bar.inc(1)
        }

        let local_timeunit = self.options.time_unit.clone();
        let self_clone = self.clone();
        let localexecutor: Arc<dyn Executor> = match self_clone.options.executor_kind {
            ExecutorKind::Raw => Arc::new(RawExecutor::new(self_clone.options)),
            ExecutorKind::Mock(ref shell) => Arc::new(MockExecutor::new(shell.clone())),
            ExecutorKind::Shell(ref shell) => {
                Arc::new(ShellExecutor::new(shell, self_clone.options))
            }
        };
        // Gather statistics (perform the actual benchmark)

        run_preparation_command()?;

        let mutex_times_real: Arc<Mutex<Vec<Second>>> = Arc::new(Mutex::new(times_real.clone()));
        let mutex_times_user: Arc<Mutex<Vec<Second>>> = Arc::new(Mutex::new(times_user.clone()));
        let mutex_times_system: Arc<Mutex<Vec<Second>>> =
            Arc::new(Mutex::new(times_system.clone()));
        let mutex_exit_codes: Arc<Mutex<Vec<Option<i32>>>> =
            Arc::new(Mutex::new(exit_codes.clone()));
        let mutex_all_succeeded: Arc<Mutex<bool>> = Arc::new(Mutex::new(all_succeeded.clone()));

        let local_command = Arc::new(self.command.deref().clone());
        let mut handles = Vec::new();
        pool =  rayon::ThreadPoolBuilder::new()
        .num_threads(self.options.num_threads as usize)
        .build()
        .unwrap();
        pool.scope(|s| {
            for cnt in 0..count_remaining {
                let t_times_real = mutex_times_real.clone();
                let t_times_user = mutex_times_user.clone();
                let t_times_system = mutex_times_system.clone();
                let t_exit_codes = mutex_exit_codes.clone();
                let t_all_succeeded = mutex_all_succeeded.clone();
                let t_command = local_command.clone();
                let t_bar = self.progress_bar.clone();
                let t_executor = localexecutor.clone();
                let handle = s.spawn(move |_| {
                    {
                        let time_temp = &t_times_real.lock().unwrap();
                        let msg = {
                            let mean = format_duration(mean(&time_temp), local_timeunit);
                            format!("Current estimate: {}", mean.to_string().green())
                        };
                        if let Some(bar) = t_bar.as_ref() {
                            bar.set_message(msg.to_owned())
                        }
                    }
                    //t_command = t_command.clone();

                    //let (res, status) = mylocalexecutor.run_command_and_measure(&thread_command, None);

                    let (res, status) = if let Ok((res, status)) =
                        t_executor.run_command_and_measure(&t_command, None)
                    {
                        (res, status)
                    } else {
                        panic!("Problem in thread {:?}: {:?}", cnt, status)
                    };
                    let success = status.success();

                    t_times_real
                        .lock()
                        .unwrap()
                        .push(res.time_real );
                    t_times_user.lock().unwrap().push(res.time_user);
                    t_times_system.lock().unwrap().push(res.time_system);

                    let mut s = t_all_succeeded.lock().unwrap();
                    *s = *s && success;

                    if let Some(bar) = t_bar.as_ref() {
                        bar.inc(1)
                    }
                    t_exit_codes.lock().unwrap().push(extract_exit_code(status));
                });
                handles.push(handle);
            }
        });

        times_real = mutex_times_real.lock().unwrap().clone();
        times_user = mutex_times_user.lock().unwrap().clone();
        times_system = mutex_times_system.lock().unwrap().clone();
        exit_codes = mutex_exit_codes.lock().unwrap().clone();

        if let Some(bar) = self.progress_bar.as_ref() {
            bar.finish()
        }

        // Compute statistical quantities
        let t_num = times_real.len();
        let t_mean = mean(&times_real);
        let t_stddev = if times_real.len() > 1 {
            Some(standard_deviation(&times_real, Some(t_mean)))
        } else {
            None
        };
        let t_median = median(&times_real);
        let t_min = min(&times_real);
        let t_max = max(&times_real);

        let user_mean = mean(&times_user);
        let system_mean = mean(&times_system);

        // Formatting and console output
        let (mean_str, time_unit) = format_duration_unit(t_mean, self.options.time_unit);
        let min_str = format_duration(t_min, Some(time_unit));
        let max_str = format_duration(t_max, Some(time_unit));
        let num_str = format!("{} runs", t_num);

        let user_str = format_duration(user_mean, Some(time_unit));
        let system_str = format_duration(system_mean, Some(time_unit));

        if self.options.output_style != OutputStyleOption::Disabled {
            if times_real.len() == 1 {
                self.multiprogress
                    .println(format!(
                        "  Time ({} ≡):        {:>8}  {:>8}     [User: {}, System: {}]",
                        "abs".green().bold(),
                        mean_str.green().bold(),
                        "        ", // alignment
                        user_str.blue(),
                        system_str.blue()
                    ))
                    .unwrap();
            } else {
                let stddev_str = format_duration(t_stddev.unwrap(), Some(time_unit));
                 self.multiprogress.println(format!("{}{}: {}",
                "Benchmark ",
                (self.number + 1),
                command_name, ))?;
                self.multiprogress
                    .println(format!(
                        "  Time ({} ± {}):     {:>8} ± {:>8}    [User: {}, System: {}]",
                        "mean".green().bold(),
                        "σ".green(),
                        mean_str.green().bold(),
                        stddev_str.green(),
                        user_str.blue(),
                        system_str.blue()
                    ))
                    .unwrap();

                self.multiprogress
                    .println(format!(
                        "  Range ({} … {}):   {:>8} … {:>8}    {} {}{:.2}",
                        "min".cyan(),
                        "max".purple(),
                        min_str.cyan(),
                        max_str.purple(),
                        num_str.dimmed(),
                        "Batch Overhead :".dimmed(),
                        batch_average,
                        //format!("{:.2}",batch_average).dimmed()
                    ))
                    .unwrap();
            }
        }

        // Warnings
        let mut warnings = vec![];

        // Check execution time
        if matches!(self.options.executor_kind, ExecutorKind::Shell(_))
            && times_real.iter().any(|&t| t < MIN_EXECUTION_TIME)
        {
            warnings.push(Warnings::FastExecutionTime);
        }

        // Check programm exit codes
        if !all_succeeded {
            warnings.push(Warnings::NonZeroExitCode);
        }

        // Run outlier detection
        let scores = modified_zscores(&times_real);
        if scores[0] > OUTLIER_THRESHOLD {
            warnings.push(Warnings::SlowInitialRun(times_real[0]));
        } else if scores.iter().any(|&s| s.abs() > OUTLIER_THRESHOLD) {
            warnings.push(Warnings::OutliersDetected);
        }

        if !warnings.is_empty() {
            self.multiprogress.println(format!(" ")).unwrap();

            for warning in &warnings {
                self.multiprogress
                    .println(format!("  {}: {}", "Warning".yellow(), warning))
                    .unwrap();
            }
        }

        if self.options.output_style != OutputStyleOption::Disabled {
            self.multiprogress.println(format!(" ")).unwrap();
        }

        self.run_cleanup_command(self.command.get_parameters().iter().cloned())?;

        Ok(BenchmarkResult {
            command: command_name,
            mean: t_mean,
            stddev: t_stddev,
            median: t_median,
            user: user_mean,
            system: system_mean,
            min: t_min,
            max: t_max,
            times: Some(times_real),
            exit_codes,
            parameters: self
                .command
                .get_parameters()
                .iter()
                .map(|(name, value)| (name.to_string(), value.to_string()))
                .collect(),
        })
    }
}
