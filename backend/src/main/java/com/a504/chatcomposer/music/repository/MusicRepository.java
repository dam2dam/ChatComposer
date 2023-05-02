package com.a504.chatcomposer.music.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.a504.chatcomposer.music.entity.Music;

@Repository
public interface MusicRepository extends JpaRepository<Music, Long> {

}
