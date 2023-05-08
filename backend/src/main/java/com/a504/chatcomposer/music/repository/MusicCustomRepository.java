package com.a504.chatcomposer.music.repository;

import java.util.List;

import org.springframework.stereotype.Repository;

import com.a504.chatcomposer.music.dto.response.MusicsResp;

/**
 * Music 관련 디비 쿼리를 정의한 인터페이스 (Querydsl 사용)
 */
@Repository
public interface MusicCustomRepository {
	List<MusicsResp> getMusicList(int genre, String tag, String nickname, String title, String isMyFavorite,
		Long loginUserId);
}
