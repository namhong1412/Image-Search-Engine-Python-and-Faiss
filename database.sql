create database data_lake;

create table image_descriptor
(
    id               bigserial     not null unique,
    path             varchar(1000) not null unique,
    color_descriptor text          not null,
    status           int           not null default 1,
    created_at       timestamp     not null default current_timestamp,
    updated_at       timestamp     not null default current_timestamp,
    primary key (id)
);