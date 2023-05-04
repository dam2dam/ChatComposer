package com.a504.chatcomposer.music.repository;

import org.springframework.data.jpa.repository.JpaRepository;

import com.a504.chatcomposer.member.entity.FavoriteMusic;

public interface FavoriteMusicRepository extends JpaRepository<FavoriteMusic, Long> {
}
