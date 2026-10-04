-- Existing installation: run once with the updated API definition in 002_api.sql.
alter table eego.workers add column if not exists user_id uuid references eego.users on delete cascade;
create index if not exists eego_worker_user_idx on eego.workers(user_id);
-- Unowned legacy tokens cannot claim user jobs.
update eego.workers set enabled=false,ready=false where user_id is null;
