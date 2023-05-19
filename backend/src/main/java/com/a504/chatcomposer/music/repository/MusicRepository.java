package com.a504.chatcomposer.music.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.a504.chatcomposer.music.entity.Music;

/**
 * Music 관련 디비 쿼리를 정의한 인터페이스 (JPA Query method 사용)
 */
@Repository
public interface MusicRepository extends JpaRepository<Music, Long>, MusicCustomRepository {

}
