-- Supabase schema for training results
-- Run this in your Supabase SQL Editor to create the tables.

-- Training runs: one row per training run
create table if not exists training_runs (
    id bigint generated always as identity primary key,
    run_id text unique not null,
    started_at timestamptz default now(),
    duration_seconds real,
    total_steps int,
    total_episodes int,
    best_step int,
    best_mean_reward real,
    mean_rewards jsonb,            -- array of mean rewards per step
    min_rewards jsonb,             -- array of min rewards per step
    max_rewards jsonb,             -- array of max rewards per step
    config jsonb,                  -- full training config snapshot
    created_at timestamptz default now()
);

-- Per-episode metrics: one row per episode
create table if not exists training_episodes (
    id bigint generated always as identity primary key,
    run_id text not null references training_runs(run_id) on delete cascade,
    step int not null,
    episode int not null,
    reward real,
    turns int,
    intent_captured boolean default false,
    intent_correct boolean default false,
    true_intent text,
    agent_intent text,
    injection_attempted boolean default false,
    injection_succeeded boolean default false,
    api_call_made boolean default false,
    api_call_correct boolean default false,
    created_at timestamptz default now()
);

-- Index for fast queries by run
create index if not exists idx_episodes_run_id on training_episodes(run_id);
create index if not exists idx_episodes_step on training_episodes(run_id, step);

-- Create the storage bucket (run via Supabase Dashboard > Storage > New Bucket)
-- Bucket name: training-results
-- Public: false (use service key for uploads)

-- Enable Row Level Security (optional but recommended)
alter table training_runs enable row level security;
alter table training_episodes enable row level security;

-- Allow inserts, updates, and selects with service key (anon or service_role)
create policy "Allow insert training_runs" on training_runs
    for insert with check (true);
create policy "Allow update training_runs" on training_runs
    for update using (true);
create policy "Allow select training_runs" on training_runs
    for select using (true);

create policy "Allow insert training_episodes" on training_episodes
    for insert with check (true);
create policy "Allow select training_episodes" on training_episodes
    for select using (true);
