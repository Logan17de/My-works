begin;

create extension if not exists pgcrypto;

create table if not exists public.strategy_levels (
  id uuid primary key default gen_random_uuid(),
  name text not null unique,
  kind text not null check (kind in ('support', 'resistance')),
  price numeric not null check (price > 0),
  source text not null default 'manual',
  enabled boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.signals (
  id uuid primary key default gen_random_uuid(),
  observed_at timestamptz not null,
  event text not null check (event in ('breakout', 'reversal', 'uncertain', 'no_level')),
  direction text not null check (direction in ('bullish', 'bearish', 'flat')),
  confidence double precision not null check (confidence >= 0 and confidence <= 1),
  combined_direction_score double precision not null check (combined_direction_score >= -1 and combined_direction_score <= 1),
  payload jsonb not null,
  created_at timestamptz not null default now()
);
create index if not exists signals_observed_at_idx on public.signals (observed_at desc);

create table if not exists public.orders (
  id uuid primary key default gen_random_uuid(),
  signal_id uuid references public.signals(id) on delete set null,
  broker_order_id text,
  mode text not null check (mode = 'paper'),
  trading_symbol text not null,
  side text not null check (side in ('BUY', 'SELL')),
  quantity integer not null check (quantity > 0),
  status text not null,
  raw jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists public.trades (
  id uuid primary key default gen_random_uuid(),
  order_id uuid references public.orders(id) on delete cascade,
  trading_symbol text not null,
  quantity integer not null check (quantity > 0),
  fill_price numeric not null check (fill_price >= 0),
  pnl numeric,
  raw jsonb not null default '{}'::jsonb,
  executed_at timestamptz not null default now()
);

alter table public.strategy_levels enable row level security;
alter table public.signals enable row level security;
alter table public.orders enable row level security;
alter table public.trades enable row level security;

drop policy if exists "dashboard can read signals" on public.signals;
create policy "dashboard can read signals" on public.signals for select to anon, authenticated using (true);

drop policy if exists "dashboard can read levels" on public.strategy_levels;
create policy "dashboard can read levels" on public.strategy_levels for select to anon, authenticated using (true);

-- orders/trades deliberately receive no anon/authenticated policies.
commit;
