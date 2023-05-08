package com.a504.chatcomposer.music.repository;

import java.util.List;

import org.springframework.stereotype.Repository;

import com.a504.chatcomposer.music.dto.response.MusicDetailResp;
import com.a504.chatcomposer.music.dto.response.MusicsResp;

/**
 * Music 관련 디비 쿼리를 정의한 인터페이스 (Querydsl 사용)
 */
@Repository
public interface MusicCustomRepository {

	/**
	 * 음악 리스트 조회
	 */
	List<MusicsResp> getMusicList(Integer genre, String tag, String nickname, String title, String isMyFavorite,
		Long loginUserId);

	/**
	 * 음악 상세 조회
	 */
	MusicDetailResp getMusicDetail(Long musicId, Long loginUserId);
}
