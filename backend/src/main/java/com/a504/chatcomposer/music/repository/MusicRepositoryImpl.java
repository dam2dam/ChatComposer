package com.a504.chatcomposer.music.repository;

import java.util.List;

import org.springframework.util.StringUtils;

import com.a504.chatcomposer.member.entity.QFavoriteMusic;
import com.a504.chatcomposer.music.dto.enums.Genre;
import com.a504.chatcomposer.music.dto.response.MusicsResp;
import com.a504.chatcomposer.music.dto.response.QMusicsResp;
import com.a504.chatcomposer.music.entity.QMusic;
import com.a504.chatcomposer.music.entity.QTag;
import com.querydsl.core.types.dsl.BooleanExpression;
import com.querydsl.jpa.impl.JPAQueryFactory;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/**
 * Music 관련 디비 쿼리를 구현한 클래스 (Querydsl 사용)
 */
@RequiredArgsConstructor
@Slf4j
public class MusicRepositoryImpl implements MusicCustomRepository {

	private static final String YES = "Y";
	private final JPAQueryFactory jpaQueryFactory;
	QMusic music = QMusic.music;
	QFavoriteMusic favoriteMusic = QFavoriteMusic.favoriteMusic;
	QTag tag = QTag.tag;

	@Override
	public List<MusicsResp> getMusicList(String genre, Long tags, Long memberId, String title, String isLiked) {

		Long userId = 0L;

		List<MusicsResp> musicsResps = jpaQueryFactory
			.select(new QMusicsResp(music, music.member.id, music.member.email))
			.from(music)
			.leftJoin(music.tags, tag)
			.leftJoin(music.favoriteMusics, favoriteMusic)
			.where(
				tagEq(tags, tag),
				// TODO: '나'의 선호음악이므로 로그인 회원 정보를 사용해야 함
				isLikedEq(isLiked, userId),
				genreEq(Genre.valueOf(genre)),
				memberEq(memberId),
				titleEq(title)
			).fetch();

		log.info("MusicRepositoryImpl | getMusicList() fetch : {}", musicsResps);
		return musicsResps;
	}

	private BooleanExpression genreEq(Genre genre) {
		return genre != null ? music.genre.eq(genre) : null;
	}

	private BooleanExpression tagEq(Long tags, QTag tag) {
		return tags != null ? tag.id.eq(tags) : null;
	}

	private BooleanExpression memberEq(Long memberId) {
		return memberId != null ? music.member.id.eq(memberId) : null;
	}

	private BooleanExpression titleEq(String title) {
		return StringUtils.hasText(title) ? music.title.contains(title) : null;
	}

	private BooleanExpression isLikedEq(String isLiked, Long userId) {
		return StringUtils.hasText(isLiked) && isLiked.equals(YES) ? favoriteMusic.member.id.eq(userId) : null;
	}
}
